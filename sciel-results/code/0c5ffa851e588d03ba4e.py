# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 250  # Safety margin for 300s limit

    X = X_train
    y = y_train
    n = len(y)
    d = X.shape[1]

    # Build rich 1D feature basis
    def build_features(X):
        x = X[:, 0]
        n = len(x)
        cols = [np.ones(n)]
        # Polynomials up to degree 8
        for p in range(1, 9):
            cols.append(x ** p)
        # Trig terms
        for k in range(1, 4):
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
        # Exp terms
        cols.append(np.exp(x))
        cols.append(np.exp(-x))
        cols.append(np.exp(2 * x))
        cols.append(np.exp(-2 * x))
        # Rational-like
        cols.append(x / (1 + x ** 2))
        cols.append(1 / (1 + x ** 2))
        # Tanh
        cols.append(np.tanh(x))
        return np.column_stack(cols)

    F_train = build_features(X)
    F_train = np.nan_to_num(F_train, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize features for better ridge behavior
    mu = F_train.mean(axis=0)
    sigma = F_train.std(axis=0)
    sigma[sigma < 1e-10] = 1.0
    F_train_std = (F_train - mu) / sigma

    # Cross-validate ridge alpha
    alphas = np.logspace(-4, 4, 19)
    kfolds = 5
    n_folds = kfolds
    fold_size = n // n_folds
    indices = np.arange(n)

    best_alpha = 1.0
    best_cv_err = np.inf

    for alpha in alphas:
        if time.time() > deadline - 20:
            break
        cv_errs = []
        for k in range(n_folds):
            val_idx = indices[k * fold_size:(k + 1) * fold_size]
            tr_idx = np.concatenate([indices[:k * fold_size], indices[(k + 1) * fold_size:]])
            F_tr = F_train_std[tr_idx]
            y_tr = y[tr_idx]
            F_val = F_train_std[val_idx]
            y_val = y[val_idx]
            # Ridge regression
            reg = alpha * np.eye(F_tr.shape[1])
            w = np.linalg.solve(F_tr.T @ F_tr + reg, F_tr.T @ y_tr)
            pred_val = F_val @ w
            cv_errs.append(np.mean((pred_val - y_val) ** 2))
        mean_err = np.mean(cv_errs)
        if mean_err < best_cv_err:
            best_cv_err = mean_err
            best_alpha = alpha

    # Final fit with best alpha
    reg = best_alpha * np.eye(F_train_std.shape[1])
    w = np.linalg.solve(F_train_std.T @ F_train_std + reg, F_train_std.T @ y)

    def predict(X):
        F = build_features(X)
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        F_std = (F - mu) / sigma
        return F_std @ w

    return predict
# EVOLVE-BLOCK-END
