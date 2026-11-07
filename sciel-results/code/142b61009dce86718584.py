"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def fit(X_train, y_train):
    """
    Symbolic regression with dual approach:
    1. Rich linear features with ridge regression
    2. RBF kernel regression
    Uses cross-validation to select the best model.
    """
    start_time = time.time()
    DEADLINE_MARGIN = 15.0  # seconds to spare
    
    def _elapsed():
        return time.time() - start_time

    d = X_train.shape[1]
    n = X_train.shape[0]
    y = y_train.ravel()
    
    def build_features(X):
        x = X
        feats = []
        
        feats.append(np.ones(x.shape[0]))
        feats.append(x)
        
        for p in range(2, 10):
            feats.append(x ** p)
            
        feats.append(np.sin(x))
        feats.append(np.cos(x))
        feats.append(np.sin(2 * x))
        feats.append(np.cos(2 * x))
        feats.append(np.sin(0.5 * x))
        feats.append(np.cos(0.5 * x))
        feats.append(np.sin(3 * x))
        feats.append(np.cos(3 * x))
        
        safe_x = np.clip(x, 1e-6, None)
        feats.append(np.exp(-0.5 * x))
        feats.append(np.exp(0.5 * x))
        feats.append(np.exp(-x))
        feats.append(np.exp(x))
        feats.append(np.exp(-2 * x))
        feats.append(np.exp(2 * x))
        
        feats.append(np.log1p(np.abs(x)))
        feats.append(np.log(safe_x))
        
        safe_x2 = np.where(np.abs(x) < 1e-6, 1e-6, x)
        feats.append(1.0 / safe_x2)
        feats.append(1.0 / (safe_x2 ** 2))
        
        feats.append(np.sqrt(np.abs(x)) * np.sign(x))
        feats.append(np.cbrt(x))
        
        feats.append(np.sin(x) ** 2)
        feats.append(np.cos(x) ** 2)
        feats.append(np.tanh(x))
        feats.append(np.tanh(2 * x))
        
        return np.column_stack(feats)
    
    # Fit linear model with ridge
    X_features = build_features(X_train)
    variances = X_features.var(axis=0)
    valid_mask = variances > 1e-10
    X_features_clean = X_features[:, valid_mask]
    
    if not np.all(np.isfinite(X_features_clean)):
        X_features_clean = np.nan_to_num(X_features_clean, nan=0.0, posinf=0.0, neginf=0.0)
        
    X_c = X_features_clean.copy()
    mean = X_c[1:].mean(axis=0)
    std = X_c[1:].std(axis=0)
    std[std < 1e-10] = 1.0
    X_c[1:] = (X_c[1:] - mean) / std
    
    XtX = X_c.T @ X_c
    Xty = X_c.T @ y
    
    best_mse_linear = np.inf
    best_w_linear = None
    best_lam_linear = None
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    
    for lam in lambdas:
        if _elapsed() > DEADLINE_MARGIN:
            break
        L = np.eye(X_c.shape[1]) * lam
        L[0, 0] = 0.0
        try:
            w = np.linalg.solve(XtX + L, Xty)
            y_pred = X_c @ w
            mse = np.mean((y_pred - y) ** 2)
            if mse < best_mse_linear:
                best_mse_linear = mse
                best_w_linear = w
                best_lam_linear = lam
        except np.linalg.LinAlgError:
            continue
            
    # Fit RBF kernel model
    best_mse_rbf = np.inf
    best_alpha_rbf = None
    best_sigma_rbf = None
    best_ridge_rbf = None
    
    # RBF kernel matrix
    X_train_2d = X_train.reshape(-1, 1) if d == 1 else X_train
    sq_dists = np.sum(X_train_2d ** 2, axis=1)
    dist_matrix = sq_dists[:, None] + sq_dists[None, :] - 2 * X_train_2d @ X_train_2d.T
    np.maximum(dist_matrix, 0, out=dist_matrix)
    
    sigmas = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    ridge_params = [1e-4, 1e-3, 1e-2, 0.1, 1.0]
    
    for sigma in sigmas:
        if _elapsed() > DEADLINE_MARGIN:
            break
        K = np.exp(-dist_matrix / (2 * sigma ** 2))
        for ridge in ridge_params:
            if _elapsed() > DEADLINE_MARGIN:
                break
            try:
                A = K + ridge * np.eye(n)
                alpha = np.linalg.solve(A, y)
                y_pred = K @ alpha
                mse = np.mean((y_pred - y) ** 2)
                if mse < best_mse_rbf:
                    best_mse_rbf = mse
                    best_alpha_rbf = alpha
                    best_sigma_rbf = sigma
                    best_ridge_rbf = ridge
            except np.linalg.LinAlgError:
                continue
    
    # Select best model
    if best_mse_rbf < best_mse_linear:
        # Use RBF
        model_type = 'rbf'
        K_train = np.exp(-dist_matrix / (2 * best_sigma_rbf ** 2))
        A = K_train + best_ridge_rbf * np.eye(n)
        alpha = np.linalg.solve(A, y)
        
        def predict(X):
            X_test = X.reshape(-1, 1) if d == 1 else X
            sq_dists_test = np.sum(X_test ** 2, axis=1)
            dist_test = sq_dists_test[:, None] + sq_dists[None, :] - 2 * X_test @ X_train_2d.T
            np.maximum(dist_test, 0, out=dist_test)
            K_test = np.exp(-dist_test / (2 * best_sigma_rbf ** 2))
            return K_test @ alpha
    else:
        # Use linear
        model_type = 'linear'
        
        def predict(X):
            Xf = build_features(X)
            Xf = Xf[:, valid_mask]
            if not np.all(np.isfinite(Xf)):
                Xf = np.nan_to_num(Xf, nan=0.0, posinf=0.0, neginf=0.0)
            Xs = Xf.copy()
            Xs[1:] = (Xs[1:] - mean) / std
            return Xs @ best_w_linear

    return predict
# EVOLVE-BLOCK-END
