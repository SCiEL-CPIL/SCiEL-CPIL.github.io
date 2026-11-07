"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 320  # safety margin before 330s wall-clock kill


def _elapsed():
    return time.time() - _DEADLINE  # positive when over


def _build_feature_matrix(X, max_degree=4):
    """Build polynomial feature matrix up to max_degree."""
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    
    # Linear terms
    for j in range(d):
        cols.append(X[:, j])
    
    # Quadratic terms
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    
    # Cubic terms
    if max_degree >= 3:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    cols.append(X[:, i] * X[:, j] * X[:, k])
    
    # Quartic terms
    if max_degree >= 4:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    for l in range(k, d):
                        cols.append(X[:, i] * X[:, j] * X[:, k] * X[:, l])
    
    return np.column_stack(cols)


def fit(X_train, y_train):
    d = X_train.shape[1]
    
    # Start with degree 2 (baseline)
    best_w = None
    best_features_fn = None
    best_mse = np.inf
    
    # Try degrees 2, 3, 4 with time budget
    for degree in [2, 3, 4]:
        if _elapsed() > 5:  # Check time budget
            break
            
        try:
            F_train = _build_feature_matrix(X_train, max_degree=degree)
            n_features = F_train.shape[1]
            
            # Use ridge regression for stability with higher degree polynomials
            # Small regularization to handle potential collinearity
            alpha = 1e-6
            F_aug = np.hstack([F_train, alpha * np.eye(n_features)])
            y_aug = np.concatenate([y_train, np.zeros(n_features)])
            
            w, *_ = np.linalg.lstsq(F_aug, y_aug, rcond=None)
            
            # Compute training MSE to compare
            pred_train = F_train @ w
            mse = np.mean((pred_train - y_train) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_w = w
                best_features_fn = lambda X, deg=degree: _build_feature_matrix(X, max_degree=deg)
                
        except Exception:
            continue
    
    # If no model was found, fall back to degree 2
    if best_w is None:
        F_train = _build_feature_matrix(X_train, max_degree=2)
        w, *_ = np.linalg.lstsq(F_train, y_train, rcond=None)
        best_w = w
        best_features_fn = lambda X: _build_feature_matrix(X, max_degree=2)
    
    w_final = best_w
    features_fn = best_features_fn
    
    def predict(X):
        F = features_fn(X)
        return F @ w_final
    
    return predict
# EVOLVE-BLOCK-END
