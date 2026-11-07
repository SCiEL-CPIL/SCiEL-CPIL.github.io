import numpy as np
import time
from scipy.linalg import solve

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin from 300s limit
    
    n, d = X_train.shape
    
    # Normalize inputs to [0, 1] for feature generation stability
    min_vals = X_train.min(axis=0)
    max_vals = X_train.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    X_norm = (X_train - min_vals) / ranges
    
    # Generate Basis Features
    # We want to capture polynomials, trig, exp, log, rational
    features = [np.ones(n)]
    
    # Polynomials up to degree 3
    for i in range(d):
        features.append(X_norm[:, i])
    for i in range(d):
        for j in range(i, d):
            features.append(X_norm[:, i] * X_norm[:, j])
    for i in range(d):
        features.append(X_norm[:, i] ** 3)
    for i in range(d):
        for j in range(i + 1, d):
            features.append(X_norm[:, i] ** 2 * X_norm[:, j])
            features.append(X_norm[:, i] * X_norm[:, j] ** 2)
            
    # Trigonometric
    for i in range(d):
        features.append(np.sin(X_norm[:, i]))
        features.append(np.cos(X_norm[:, i]))
        
    # Exponential and Logarithmic (using normalized inputs to keep scale manageable)
    for i in range(d):
        features.append(np.exp(-X_norm[:, i]))
        features.append(np.log(X_norm[:, i] + 0.1))
        features.append(1.0 / (X_norm[:, i] + 0.1))
        
    # Combine features
    X_feat = np.column_stack(features)
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Split for validation
    n_val = int(n * 0.2)
    val_idx = np.arange(n - n_val, n)
    train_idx = np.arange(0, n - n_val)
    
    X_tr_feat = X_feat[train_idx]
    y_tr = y_train[train_idx]
    X_val_feat = X_feat[val_idx]
    y_val = y_train[val_idx]
    
    best_model_type = None
    best_params = None
    best_val_mse = np.inf
    
    # 1. Try Ridge Regression on the basis
    alphas_ridge = [1e-6, 1e-4, 1e-2, 1e0]
    
    for alpha in alphas_ridge:
        if time.time() > deadline:
            break
        try:
            XtX = X_tr_feat.T @ X_tr_feat
            Xty = X_tr_feat.T @ y_tr
            w = solve(XtX + alpha * np.eye(XtX.shape[0]), Xty, assume_a='pos')
            
            y_pred_val = X_val_feat @ w
            mse = np.mean((y_pred_val - y_val) ** 2)
            
            if mse < best_val_mse:
                best_val_mse = mse
                best_model_type = 'ridge'
                best_params = {'alpha': alpha}
        except:
            pass
            
    # 2. Try GP with Matern kernel (better for non-polynomial smooth functions)
    X_tr_norm = X_norm[train_idx]
    X_val_norm = X_norm[val_idx]
    
    def compute_kernel_matern(X1, X2, length_scale, nu=1.5):
        # Matern kernel
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        dists = np.sqrt(sq_dists)
        
        if nu == 1.5:
            K = np.exp(-np.sqrt(3) * dists / length_scale) * (1 + np.sqrt(3) * dists / length_scale)
        elif nu == 2.5:
            K = np.exp(-np.sqrt(5) * dists / length_scale) * (1 + np.sqrt(5) * dists / dists + 5/3 * (dists/length_scale)**2)
        else:
            # Fallback to RBF
            K = np.exp(-0.5 * dists**2 / length_scale**2)
        
        return K + 1e-8 * np.eye(len(X1)) if len(X1) == len(X2) else K

    # Try different length scales for Matern kernel
    length_scales = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    
    for ls in length_scales:
        if time.time() > deadline:
            break
        try:
            K_tr = compute_kernel_matern(X_tr_norm, X_tr_norm, ls)
            K_val = compute_kernel_matern(X_val_norm, X_tr_norm, ls)
            
            w = solve(K_tr, y_tr, assume_a='pos')
            y_pred_val = K_val @ w
            mse = np.mean((y_pred_val - y_val) ** 2)
            
            if mse < best_val_mse:
                best_val_mse = mse
                best_model_type = 'gp_matern'
                best_params = {'length_scale': ls}
        except:
            pass

    # If no model was found, default to GP with moderate length scale
    if best_model_type is None:
        best_model_type = 'gp_matern'
        best_params = {'length_scale': 0.5}
        
    # Final Fit on all data with best parameters
    if best_model_type == 'ridge':
        alpha = best_params['alpha']
        XtX = X_feat.T @ X_feat
        Xty = X_feat.T @ y_train
        w = solve(XtX + alpha * np.eye(XtX.shape[0]), Xty, assume_a='pos')
        
        def predict(X):
            X_norm_test = (X - min_vals) / ranges
            feats_test = [np.ones(len(X))]
            for i in range(d):
                feats_test.append(X_norm_test[:, i])
            for i in range(d):
                for j in range(i, d):
                    feats_test.append(X_norm_test[:, i] * X_norm_test[:, j])
            for i in range(d):
                feats_test.append(X_norm_test[:, i] ** 3)
            for i in range(d):
                for j in range(i + 1, d):
                    feats_test.append(X_norm_test[:, i] ** 2 * X_norm_test[:, j])
                    feats_test.append(X_norm_test[:, i] * X_norm_test[:, j] ** 2)
            for i in range(d):
                feats_test.append(np.sin(X_norm_test[:, i]))
                feats_test.append(np.cos(X_norm_test[:, i]))
            for i in range(d):
                feats_test.append(np.exp(-X_norm_test[:, i]))
                feats_test.append(np.log(X_norm_test[:, i] + 0.1))
                feats_test.append(1.0 / (X_norm_test[:, i] + 0.1))
                
            X_feat_test = np.column_stack(feats_test)
            X_feat_test = np.nan_to_num(X_feat_test, nan=0.0, posinf=1e6, neginf=-1e6)
            return X_feat_test @ w

    else: # GP with Matern
        length_scale = best_params['length_scale']
        K_train = compute_kernel_matern(X_norm, X_norm, length_scale)
        w = solve(K_train, y_train, assume_a='pos')
        
        def predict(X):
            X_norm_test = (X - min_vals) / ranges
            K_test = compute_kernel_matern(X_norm_test, X_norm, length_scale)
            return K_test @ w

    return predict
