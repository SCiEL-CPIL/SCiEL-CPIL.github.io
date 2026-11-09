"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_features(X):
    """Build a rich set of features: polynomials up to degree 4, trig, exp, reciprocal, cross-terms."""
    n = X.shape[0]
    d = X.shape[1]
    cols = [np.ones(n)]
    
    # Individual powers up to degree 4
    for j in range(d):
        xj = X[:, j]
        cols.append(xj)
        cols.append(xj ** 2)
        cols.append(xj ** 3)
        cols.append(xj ** 4)
        # Trig features
        cols.append(np.sin(xj))
        cols.append(np.cos(xj))
        # Exponential (careful with large values)
        cols.append(np.exp(-xj))
        # Reciprocal
        cols.append(1.0 / (xj + 1e-8))
    
    # Pairwise products (degree 2 cross terms)
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    
    # Triple products for small d
    if d <= 3:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    cols.append(X[:, i] * X[:, j] * X[:, k])
    
    # Sum and product of all inputs
    if d >= 2:
        cols.append(np.sum(X, axis=1))
        cols.append(np.prod(X, axis=1))
    
    return np.column_stack(cols)


def fit(X_train, y_train):
    t0 = time.time()
    deadline = t0 + 280.0  # Leave margin for 330s total across all equations
    
    F_train = _build_features(X_train)
    
    # Use ridge regression for stability with many features
    n_features = F_train.shape[1]
    
    # Try increasing lambda values, pick best on training data
    best_w = None
    best_mse = np.inf
    
    for lam in [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
        if time.time() > deadline:
            break
        try:
            A = F_train.T @ F_train + lam * np.eye(n_features)
            b = F_train.T @ y_train
            w = np.linalg.solve(A, b)
            pred = F_train @ w
            mse = np.mean((pred - y_train) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_w = w
        except np.linalg.LinAlgError:
            continue
    
    if best_w is None:
        # Fallback to lstsq
        best_w, *_ = np.linalg.lstsq(F_train, y_train, rcond=None)
    
    w_final = best_w.copy()
    
    def predict(X):
        F = _build_features(X)
        return F @ w_final
    
    return predict
# EVOLVE-BLOCK-END
