"""Symbolic regression over 10 hidden equations."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = start_time + 270.0  # leave 60s safety margin out of 330s

    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)

    def predict_const(X):
        return np.full(len(X), y_mean)

    if y_var < 1e-14:
        return predict_const

    def time_ok():
        return time.time() < DEADLINE

    # Build a rich feature matrix: polynomials up to degree 3, trig, exp, log, sqrt
    def build_features(X, max_deg=3):
        cols = [np.ones(len(X))]
        # Linear terms
        for j in range(d):
            cols.append(X[:, j])
        # Quadratic terms
        for i in range(d):
            for j in range(i, d):
                cols.append(X[:, i] * X[:, j])
        # Cubic terms
        if max_deg >= 3:
            for i in range(d):
                for j in range(i, d):
                    for k in range(j, d):
                        cols.append(X[:, i] * X[:, j] * X[:, k])
        # Trig features
        for j in range(d):
            cols.append(np.sin(X[:, j]))
            cols.append(np.cos(X[:, j]))
        # Exp and log (safe)
        for j in range(d):
            cols.append(np.exp(np.clip(X[:, j], -10, 10)))
            cols.append(np.log1p(np.abs(X[:, j])))
            cols.append(np.sqrt(np.abs(X[:, j])))
        # Interactions with trig
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(np.sin(X[:, i]) * np.cos(X[:, j]))
                cols.append(np.cos(X[:, i]) * np.sin(X[:, j]))
        return np.column_stack(cols)

    # Try multiple feature sets and pick the best via cross-validation on train
    def cv_score(feat_fn, max_deg=3):
        """Simple 5-fold CV MSE."""
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        fold_size = n // 5
        mses = []
        for fold in range(5):
            val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            tr_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            F_tr = feat_fn(X_tr, max_deg)
            F_val = feat_fn(X_val, max_deg)
            # Regularized least squares
            lam = 1e-3
            w = np.linalg.solve(F_tr.T @ F_tr + lam * np.eye(F_tr.shape[1]), F_tr.T @ y_tr)
            pred = F_val @ w
            mse = np.mean((pred - y_val) ** 2)
            mses.append(mse)
        return np.mean(mses)

    # Feature builders with different max degrees
    best_score = np.inf
    best_w = None
    best_feat_fn = None

    for max_deg in [2, 3]:
        if not time_ok():
            break
        feat_fn = lambda X, md=max_deg: build_features(X, md)
        # First check if feature matrix is well-conditioned
        F_train = feat_fn(X_train)
        if F_train.shape[1] > n:
            # Too many features, use ridge
            pass
        try:
            score = cv_score(feat_fn, max_deg)
            if score < best_score:
                best_score = score
                best_feat_fn = feat_fn
                # Fit final model
                F_train = feat_fn(X_train)
                lam = 1e-3
                best_w = np.linalg.solve(F_train.T @ F_train + lam * np.eye(F_train.shape[1]), F_train.T @ y_train)
        except Exception:
            continue

    # If no model worked, fall back to simple linear
    if best_w is None:
        F_train = build_features(X_train, 2)
        try:
            lam = 1e-3
            best_w = np.linalg.solve(F_train.T @ F_train + lam * np.eye(F_train.shape[1]), F_train.T @ y_train)
            best_feat_fn = lambda X: build_features(X, 2)
        except Exception:
            return predict_const

    # Also try a simpler linear model as comparison
    try:
        F_lin = np.column_stack([np.ones(n), X_train])
        lam = 1e-4
        w_lin = np.linalg.solve(F_lin.T @ F_lin + lam * np.eye(F_lin.shape[1]), F_lin.T @ y_train)
        # CV for linear
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        fold_size = n // 5
        mses = []
        for fold in range(5):
            val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            tr_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            F_tr = np.column_stack([np.ones(len(X_tr)), X_tr])
            F_val = np.column_stack([np.ones(len(X_val)), X_val])
            w = np.linalg.solve(F_tr.T @ F_tr + lam * np.eye(F_tr.shape[1]), F_tr.T @ y_tr)
            pred = F_val @ w
            mses.append(np.mean((pred - y_val) ** 2))
        lin_score = np.mean(mses)
        if lin_score < best_score:
            best_w = w_lin
            best_feat_fn = lambda X: np.column_stack([np.ones(len(X)), X])
            best_score = lin_score
    except Exception:
        pass

    def predict(X):
        F = best_feat_fn(X)
        return F @ best_w

    return predict
# EVOLVE-BLOCK-END
