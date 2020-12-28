# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:16:01 2020

@author: shara
"""

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

salary_train = pd.read_csv("F:\\Warun\\DS Assignments\\DS Assignments\\Naive Bayes\\SalaryData_Train.csv")
salary_test = pd.read_csv("F:\\Warun\\DS Assignments\\DS Assignments\\Naive Bayes\\SalaryData_Test.csv")
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

for col in string_columns:
    plt.figure(figsize = (11,6))
    sns.barplot(salary_train[col].value_counts(),salary_train[col].value_counts().index, data = salary_train)
    plt.title(col)
    plt.tight_layout()


from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
    salary_train[i] = number.fit_transform(salary_train[i])
    salary_test[i] = number.fit_transform(salary_test[i])

column_list = []
iqr_list = []
out_low = []
out_up = []
tot_outlier = []

for i in salary_train.describe().columns : 
    QTR1 = salary_train[i].quantile(0.25)
    QTR3 = salary_train[i].quantile(0.75)
    IQR = QTR3 - QTR1
    LTV = QTR1 - (1.5* IQR)
    UTV = QTR3 + (1.5 * IQR)
    current_column = i
    current_iqr = IQR
    bl_LTV = salary_train[salary_train[i] < LTV][i].count()
    ab_UTV = salary_train[salary_train[i] > UTV][i].count()
    TOT_outliers = bl_LTV + ab_UTV
    column_list.append(current_column)
    iqr_list.append(current_iqr)
    out_low.append(bl_LTV)
    out_up.append(ab_UTV)
    tot_outlier.append(TOT_outliers)
    outlier_report = {"Column_name" : column_list, "IQR" : iqr_list, "Below_outliers" : out_low, "Above_outlier" : out_up, "Total_outliers" : tot_outlier}
    outlier_report = pd.DataFrame(outlier_report)
    print(outlier_report)

sns.boxplot(data = salary_train.age , orient = "n", palette = "Set3")
sns.boxplot(data = salary_train.workclass , orient = "n", palette = "Set3")
sns.boxplot(data = salary_train.education , orient = "n", palette = "Set3")
sns.boxplot(data = salary_train.educationno , orient = "n", palette = "Set3")
sns.boxplot(data = salary_train.maritalstatus , orient = "n", palette = "Set3")
sns.boxplot(data = salary_train.occupation , orient = "n", palette = "Set3")
sns.boxplot(data = salary_train.relationship , orient = "n", palette = "Set3")
sns.boxplot(data = salary_train.race , orient = "n", palette = "Set3")
sns.boxplot(data = salary_train.sex , orient = "n", palette = "Set3")
sns.boxplot(data = salary_train.capitalgain , orient = "n", palette = "Set3")
sns.boxplot(data = salary_train.capitalloss , orient = "n", palette = "Set3")
sns.boxplot(data = salary_train.hoursperweek , orient = "n", palette = "Set3")
sns.boxplot(data = salary_train.native , orient = "n", palette = "Set3")

f, ax = plt.subplots(figsize=(12,6))
sns.heatmap(salary_train.corr(), annot=True, fmt='.2f')

colnames = salary_train.columns
len(colnames[0:13])
trainX = salary_train[colnames[0:13]]
trainY = salary_train[colnames[13]]
testX  = salary_test[colnames[0:13]]
testY  = salary_test[colnames[13]]

sgnb = GaussianNB()
smnb = MultinomialNB()
spred_gnb = sgnb.fit(trainX,trainY).predict(trainX)
confusion_matrix(trainY,spred_gnb)
print ("Accuracy",(21505+2483)/(21505+2483+5025+1148)) 

spred_mnb = smnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_mnb)
print("Accuracy",(10891+780)/(10891+780+2920+780))

