import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import seaborn as sns


data=pd.read_csv(r"E:\Machine Learning Projects\Online Payments Fraud Detection with Machine Learning\Data\Data\PS_20174392719_1491204439457_log.csv")

print(data.head())
print(data.describe())

corr=data.corr()
sns.heatmap(corr)

print(corr["isFraud"].sort_values(ascending=False))
data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
print(data.head())


from sklearn.model_selection import train_test_split
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.10,random_state=42)
model1=DecisionTreeClassifier()
model1.fit(xtrain,ytrain)
print(model1.score(xtest, ytest))

model2=LogisticRegression()
model2.fit(xtrain,ytrain)
print(model2.score(xtest, ytest))




