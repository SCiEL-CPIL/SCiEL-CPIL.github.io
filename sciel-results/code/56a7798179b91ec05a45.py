"""Symbolic regression over 10 hidden equations via adaptive feature regression with time guard."""
import numpy as np
import time
from math import log, exp, sqrt

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 310.0  # safety margin before 330s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Check if y is constant
    y_mean = np.mean(y)
    if np.allclose(y, y_mean, atol=1e-12):
        return lambda X_: np.full(len(X_), y_mean)

    # Build a rich feature set
    def build_features(X):
        cols = [np.ones(len(X))]
        # Linear
        for j in range(d):
            cols.append(X[:, j])
        # Quadratic
        for i in range(d):
            for j in range(i, d):
                cols.append(X[:, i] * X[:, j])
        # Cubic (only for small d to keep feature count manageable)
        if d <= 3:
            for i in range(d):
                for j in range(i, d):
                    for k in range(j, d):
                        cols.append(X[:, i] * X[:, j] * X[:, k])
        # Non-linear basis functions
        for j in range(d):
            xj = X[:, j]
            cols.append(np.sin(xj))
            cols.append(np.cos(xj))
            cols.append(np.exp(-xj))  # decaying exp
            cols.append(np.log(xj + 1e-8))  # safe log
            cols.append(np.sqrt(np.maximum(xj, 0)))
            cols.append(1.0 / (xj + 1e-8))  # reciprocal
            cols.append(xj * np.sin(xj))
            cols.append(xj * np.cos(xj))
        # Cross non-linear terms (limited)
        if d >= 2:
            for i in range(d):
                for j in range(i + 1, d):
                    cols.append(np.sin(X[:, i]) * np.cos(X[:, j]))
                    cols.append(np.cos(X[:, i]) * np.sin(X[:, j]))
                    cols.append(X[:, i] * np.sin(X[:, j]))
                    cols.append(np.sin(X[:, i]) * X[:, j])
        return np.column_stack(cols)

    # Compute features
    F_train = build_features(X)

    # Handle potential NaN/Inf from non-linear functions
    F_train = np.nan_to_num(F_train, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize features for better conditioning
    F_mean = np.mean(F_train, axis=0)
    F_std = np.std(F_train, axis=0)
    F_std[F_std < 1e-10] = 1.0
    F_train_std = (F_train - F_mean) / F_std

    # Ridge regression with multiple lambda values, pick best via validation
    # Use a simple train/test split for model selection
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.8 * n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    y_train_sub = y[train_idx]
    y_val_sub = y[val_idx]
    F_train_sub = F_train_std[train_idx]
    F_val_sub = F_train_std[val_idx]

    # Try several regularization strengths
    best_lambda = 1.0
    best_val_mse = np.inf
    best_w = None

    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]

    for lam in lambdas:
        if time.time() > DEADLINE:
            break
        # Ridge: (F^T F + lam*I) w = F^T y
        FtF = F_train_sub.T @ F_train_sub
        Fty = F_train_sub.T @ y_train_sub
        A = FtF + lam * np.eye(F_train_sub.shape[1])
        try:
            w = np.linalg.solve(A, Fty)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, Fty, rcond=None)[0]
        
        pred_val = F_val_sub @ w
        mse = np.mean((pred_val - y_val_sub) ** 2)
        if mse < best_val_mse:
            best_val_mse = mse
            best_lambda = lam
            best_w = w

    # Refit with best lambda on all data
    FtF = F_train_std.T @ F_train_std
    Fty = F_train_std.T @ y
    A = FtF + best_lambda * np.eye(F_train_std.shape[1])
    try:
        w_final = np.linalg.solve(A, Fty)
    except np.linalg.LinAlgError:
        w_final = np.linalg.lstsq(A, Fty, rcond=None)[0]

    # If time remains, try a higher-degree polynomial expansion as an alternative
    # and compare on the validation set
    alt_w = None
    if time.time() < DEADLINE - 10:
        def build_features_high(X):
            cols = [np.ones(len(X))]
            for j in range(d):
                cols.append(X[:, j])
            for i in range(d):
                for j in range(i, d):
                    cols.append(X[:, i] * X[:, j])
            if d <= 3:
                for i in range(d):
                    for j in range(i, d):
                        for k in range(j, d):
                            cols.append(X[:, i] * X[:, j] * X[:, k])
            # Add more non-linear
            for j in range(d):
                xj = X[:, j]
                cols.append(np.sin(2 * xj))
                cols.append(np.cos(2 * xj))
                cols.append(np.exp(-2 * xj))
                cols.append(np.log(xj + 1e-8))
                cols.append(np.sqrt(np.maximum(xj, 0)))
                cols.append(1.0 / (xj + 1e-8))
                cols.append(xj ** 3)
            return np.column_stack(cols)

        F_train_alt = build_features_high(X)
        F_train_alt = np.nan_to_num(F_train_alt, nan=0.0, posinf=0.0, neginf=0.0)
        F_mean_alt = np.mean(F_train_alt, axis=0)
        F_std_alt = np.std(F_train_alt, axis=0)
        F_std_alt[F_std_alt < 1e-10] = 1.0
        F_train_alt_std = (F_train_alt - F_mean_alt) / F_std_alt

        # Use same best_lambda for consistency
        FtF_alt = F_train_alt_std[train_idx].T @ F_train_alt_std[train_idx]
        Fty_alt = F_train_alt_std[train_idx].T @ y_train_sub
        A_alt = FtF_alt + best_lambda * np.eye(F_train_alt_std.shape[1])
        try:
            w_alt = np.linalg.solve(A_alt, Fty_alt)
        except np.linalg.LinAlgError:
            w_alt = np.linalg.lstsq(A_alt, Fty_alt, rcond=None)[0]

        pred_val_alt = F_train_alt_std[val_idx] @ w_alt
        mse_alt = np.mean((pred_val_alt - y_val_sub) ** 2)

        if mse_alt < best_val_mse:
            # Use the alternative
            F_mean_final = F_mean_alt
            F_std_final = F_std_alt
            w_final = w_alt
            F_train_final = F_train_alt_std
            use_alt = True
        else:
            F_mean_final = F_mean
            F_std_final = F_std
            use_alt = False

    # Final feature builder matching the chosen model
    if use_alt:
        def _build_feats(X_):
            cols = [np.ones(len(X_))]
            for j in range(d):
                cols.append(X_[:, j])
            for i in range(d):
                for j in range(i, d):
                    cols.append(X_[:, i] * X_[:, j])
            if d <= 3:
                for i in range(d):
                    for j in range(i, d):
                        for k in range(j, d):
                            cols.append(X_[:, i] * X_[:, j] * X_[:, k])
            for j in range(d):
                xj = X_[:, j]
                cols.append(np.sin(2 * xj))
                cols.append(np.cos(2 * xj))
                cols.append(np.exp(-2 * xj))
                cols.append(np.log(xj + 1e-8))
                cols.append(np.sqrt(np.maximum(xj, 0)))
                cols.append(1.0 / (xj + 1e-8))
                cols.append(xj ** 3)
            return np.column_stack(cols)
    else:
        def _build_feats(X_):
            cols = [np.ones(len(X_))]
            for j in range(d):
                cols.append(X_[:, j])
            for i in range(d):
                for j in range(i, d):
                    cols.append(X_[:, i] * X_[:, j])
            if d <= 3:
                for i in range(d):
                    for j in range(i, d):
                        for k in range(j, d):
                            cols.append(X_[:, i] * X_[:, j] * X_[:, k])
            for j in range(d):
                xj = X_[:, j]
                cols.append(np.sin(xj))
                cols.append(np.cos(xj))
                cols.append(np.exp(-xj))
                cols.append(np.log(xj + 1e-8))
                cols.append(np.sqrt(np.maximum(xj, 0)))
                cols.append(1.0 / (xj + 1e-8))
                cols.append(xj * np.sin(xj))
                cols.append(xj * np.cos(xj))
            if d >= 2:
                for i in range(d):
                    for j in range(i + 1, d):
                        cols.append(np.sin(X_[:, i]) * np.cos(X_[:, j]))
                        cols.append(np.cos(X_[:, i]) * np.sin(X_[:, j]))
                        cols.append(X_[:, i] * np.sin(X_[:, j]))
                        cols.append(np.sin(X_[:, i]) * X_[:, j])
            return np.column_stack(cols)

    w_use = w_final
    mean_use = F_mean_final if use_alt else F_mean
    std_use = F_std_final if use_alt else F_std

    def predict(X_):
        X_ = np.asarray(X_, dtype=np.float64)
        F_ = _build_feats(X_)
        F_ = np.nan_to_num(F_, nan=0.0, posinf=0.0, neginf=0.0)
        F_std_ = (F_ - mean_use) / std_use
        return F_std_ @ w_use

    return predict
