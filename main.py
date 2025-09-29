"""
Heart Disease Prediction - Decision Tree & Random Forest
Works with a LOCAL dataset (CSV).
Replace the 'path' variable with the location of your heart.csv file.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ---------- Config ----------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV = 5

# ---------- Load dataset ----------
def load_dataset():
    # 🔹 CHANGE this path to where your CSV file is saved
    path = r"C:\Users\VEERESH AWARALLI\Downloads\heart.csv"
    df = pd.read_csv(path)
    print(f"✅ Dataset loaded successfully from: {path}")
    return df

# ---------- Preprocess ----------
def preprocess(df):
    df.columns = [c.strip() for c in df.columns]

    # Identify target column
    if 'target' in df.columns:
        target_col = 'target'
    elif 'num' in df.columns:
        target_col = 'num'
    else:
        target_col = df.columns[-1]  # fallback
    print("🎯 Using target column:", target_col)

    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # Handle missing values
    df.replace('?', np.nan, inplace=True)
    for c in X.columns:
        if X[c].dtype.kind in 'biufc':  # numeric
            X[c] = pd.to_numeric(X[c], errors='coerce')
            X[c].fillna(X[c].median(), inplace=True)
        else:  # categorical
            X[c].fillna(X[c].mode()[0], inplace=True)

    # Encode categorical features
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Encode target if categorical
    if y.dtype == 'object' or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y)

    return X, y, target_col

# ---------- Train & Evaluate ----------
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    print("\n🌳 Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print("🌲 Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

    # Cross-validation
    cv_dt = cross_val_score(dt, X, y, cv=CV).mean()
    cv_rf = cross_val_score(rf, X, y, cv=CV).mean()
    print(f"\n🔁 CV Accuracy - Decision Tree: {cv_dt:.4f}")
    print(f"🔁 CV Accuracy - Random Forest: {cv_rf:.4f}")

    # Reports
    print("\n📊 Classification Report - Decision Tree")
    print(classification_report(y_test, y_pred_dt))

    print("\n📊 Classification Report - Random Forest")
    print(classification_report(y_test, y_pred_rf))

    print("\n📊 Confusion Matrix - Random Forest")
    print(confusion_matrix(y_test, y_pred_rf))

    # Feature Importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 5))
    plt.title("Feature Importances (Random Forest)")
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()
    plt.show()

    # Decision Tree Plot
    plt.figure(figsize=(16, 8))
    plot_tree(
        dt,
        feature_names=X.columns,
        class_names=[str(c) for c in np.unique(y)],
        filled=True,
        rounded=True,
        fontsize=8
    )
    plt.show()

# ---------- Main ----------
def main():
    df = load_dataset()
    print("📂 Dataset shape:", df.shape)
    X, y, target_col = preprocess(df)
    print("✅ Features:", X.shape, "Target:", y.shape)
    train_and_evaluate(X, y)

if __name__ == "__main__":
    main()
