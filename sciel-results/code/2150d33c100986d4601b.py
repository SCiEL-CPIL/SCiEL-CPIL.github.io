"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 250  # Safety margin for 300s limit

    X = X_train
    y = y_train
    d = X.shape[1]
    n = len(y)

    def features(X):
        n = len(X)
        cols = [np.ones(n)]
        # Linear terms
        for j in range(d):
            cols.append(X[:, j])
        # Quadratic terms
        for i in range(d):
            for j in range(i, d):
                cols.append(X[:, i] * X[:, j])
        # Cubic terms
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    cols.append(X[:, i] * X[:, j] * X[:, k])
        # Quartic terms (only for 1D to keep it manageable)
        if d == 1:
            x = X[:, 0]
            cols.append(x ** 4)
            # Trig features
            cols.append(np.sin(x))
            cols.append(np.cos(x))
            cols.append(np.sin(2 * x))
            cols.append(np.cos(2 * x))
            # Reciprocal and sqrt (clip to avoid issues)
            x_safe = np.where(np.abs(x) > 1e-6, x, np.sign(x) * 1e-6)
            cols.append(1.0 / x_safe)
            cols.append(np.sqrt(np.abs(x)))
            # Exponential
            cols.append(np.exp(np.clip(x, -10, 10)))
        elif d > 1:
            for j in range(d):
                x = X[:, j]
                cols.append(np.sin(x))
                cols.append(np.cos(x))
        return np.column_stack(cols)

    F_train = features(X)
    
    # Use ridge regression for stability
    # Try a few regularization strengths
    best_pred = None
    best_train_mse = np.inf
    
    for alpha in [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]:
        if time.time() > deadline - 5:
            break
        try:
            if alpha == 0.0:
                w, *_ = np.linalg.lstsq(F_train, y, rcond=None)
            else:
                w = np.linalg.solve(F_train.T @ F_train + alpha * np.eye(F_train.shape[1]), F_train.T @ y)
            
            pred_train = F_train @ w
            mse = np.mean((pred_train - y) ** 2)
            if mse < best_train_mse:
                best_train_mse = mse
                best_w = w
        except:
            continue
    
    if best_w is None:
        # Fallback to simple linear
        w, *_ = np.linalg.lstsq(F_train, y, rcond=None)
        best_w = w

    def predict(X_test):
        F_test = features(X_test)
        pred = F_test @ best_w
        # Ensure finite values
        pred = np.where(np.isfinite(pred), pred, 0.0)
        return pred

    return predict
# EVOLVE-BLOCK-END
