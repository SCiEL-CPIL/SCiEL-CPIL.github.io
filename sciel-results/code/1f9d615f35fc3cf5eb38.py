import numpy as np
import time
from scipy.linalg import solve
from scipy.optimize import minimize
import warnings

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 290.0  # Safety margin from 300s limit (evaluator kills at 330s)
    
    n, d = X_train.shape
    
    # Normalize inputs to [0, 1] for feature generation stability
    min_vals = X_train.min(axis=0)
    max_vals = X_train.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    X_norm = (X_train - min_vals) / ranges
    
    # Center y for numerical stability in regression
    y_mean = np.mean(y_train)
    y_centered = y_train - y_mean
    
    # Split for validation
    n_val = max(10, int(n * 0.2))
    val_idx = np.arange(n - n_val, n)
    train_idx = np.arange(0, n - n_val)
    
    X_tr_norm = X_norm[train_idx]
    y_tr = y_centered[train_idx]
    X_val_norm = X_norm[val_idx]
    y_val = y_centered[val_idx]
    
    def generate_polynomial_features(X, degree):
        """Generate polynomial features up to 'degree' for input X."""
        n = X.shape[0]
        d = X.shape[1]
        feats = [np.ones(n)]
        
        # Generate all monomials x1^a1 * x2^a2 * ... * xd^ad where sum(ai) <= degree
        # We can do this recursively or by iterating over combinations with replacement
        # For small d (<=3) and degree (<=4), explicit loops are fast enough.
        
        # Degree 1
        if degree >= 1:
            for i in range(d):
                feats.append(X[:, i])
        
        # Degree 2
        if degree >= 2:
            for i in range(d):
                for j in range(i, d):
                    feats.append(X[:, i] * X[:, j])
        
        # Degree 3
        if degree >= 3:
            for i in range(d):
                for j in range(i, d):
                    for k in range(j, d):
                        feats.append(X[:, i] * X[:, j] * X[:, k])
                        
        # Degree 4
        if degree >= 4:
            for i in range(d):
                for j in range(i, d):
                    for k in range(j, d):
                        for l in range(k, d):
                            feats.append(X[:, i] * X[:, j] * X[:, k] * X[:, l])
                            
        # Degree 5 (only if d is small, e.g., d=1)
        if degree >= 5 and d == 1:
            feats.append(X[:, 0] ** 5)
            
        return np.column_stack(feats)

    def rbf_kernel(X1, X2, length_scale, signal_var):
        """Compute RBF kernel with given hyperparameters."""
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        K = signal_var * np.exp(-sq_dists / (2 * length_scale**2))
        if len(X1) == len(X2):
            K += 1e-8 * np.eye(len(X1))
        return K

    # List of candidate models to evaluate
    # Each model is a tuple: (name, feature_generator, alpha_ridge)
    # We will test Ridge Regression with different polynomial degrees and regularization strengths.
    candidates = []
    
    # Ridge Regression candidates
    # Degrees 1 to 4 are usually sufficient for "hidden equations" in these tasks.
    for deg in [1, 2, 3, 4]:
        # Try a few alpha values for regularization
        for alpha in [1e-4, 1e-2, 1.0, 10.0]:
            candidates.append((f"Ridge_D{deg}_A{alpha}", deg, alpha))
            
    # GP Candidate
    # We'll include a GP with fixed, reasonable hyperparameters as a fallback.
    # If the function is non-polynomial (e.g., sin, exp), GP should handle it better.
    candidates.append(("GP_Fixed", "gp", 0.0))

    best_model_name = None
    best_val_mse = np.inf
    best_model_data = None

    for name, deg_or_type, alpha in candidates:
        if time.time() > deadline - 10:
            break
            
        try:
            if deg_or_type == "gp":
                # Lightweight GP with fixed hyperparameters
                # These are reasonable defaults for normalized [0,1] inputs
                length_scale = 0.3
                signal_var = 1.0
                noise_var = 1e-4
                
                K_tr_tr = rbf_kernel(X_tr_norm, X_tr_norm, length_scale, signal_var) + noise_var * np.eye(len(X_tr_norm))
                
                try:
                    w = solve(K_tr_tr, y_tr, assume_a='pos')
                except:
                    w = np.linalg.lstsq(K_tr_tr, y_tr, rcond=None)[0]
                    
                K_val_tr = rbf_kernel(X_val_norm, X_tr_norm, length_scale, signal_var)
                y_pred_val = K_val_tr @ w
                val_mse = np.mean((y_pred_val - y_val) ** 2)
                
                if np.isfinite(val_mse) and val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_model_name = name
                    # Store data needed to refit on full set
                    best_model_data = {
                        'type': 'gp',
                        'length_scale': length_scale,
                        'signal_var': signal_var,
                        'noise_var': noise_var
                    }
            else:
                # Ridge Regression
                X_tr_feat = generate_polynomial_features(X_tr_norm, deg_or_type)
                X_val_feat = generate_polynomial_features(X_val_norm, deg_or_type)
                
                # Solve (X^T X + alpha I) w = X^T y
                A = X_tr_feat.T @ X_tr_feat + alpha * np.eye(X_tr_feat.shape[1])
                b = X_tr_feat.T @ y_tr
                
                try:
                    w = solve(A, b, assume_a='pos')
                except:
                    w = np.linalg.lstsq(A, b, rcond=None)[0]
                    
                y_pred_val = X_val_feat @ w
                val_mse = np.mean((y_pred_val - y_val) ** 2)
                
                if np.isfinite(val_mse) and val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_model_name = name
                    best_model_data = {
                        'type': 'ridge',
                        'degree': deg_or_type,
                        'alpha': alpha
                    }
                    
        except Exception as e:
            # Skip if this model fails
            continue

    # If no model was selected (should not happen), default to simple linear
    if best_model_data is None:
        best_model_data = {'type': 'ridge', 'degree': 1, 'alpha': 1.0}
        
    # Refit the best model on ALL training data
    if best_model_data['type'] == 'ridge':
        degree = best_model_data['degree']
        alpha = best_model_data['alpha']
        
        X_all_feat = generate_polynomial_features(X_norm, degree)
        A_full = X_all_feat.T @ X_all_feat + alpha * np.eye(X_all_feat.shape[1])
        b_full = X_all_feat.T @ y_centered
        
        try:
            w_full = solve(A_full, b_full, assume_a='pos')
        except:
            w_full = np.linalg.lstsq(A_full, b_full, rcond=None)[0]
            
        def predict(X):
            X_test_norm = (X - min_vals) / ranges
            X_test_feat = generate_polynomial_features(X_test_norm, degree)
            return X_test_feat @ w_full + y_mean

    else: # GP
        length_scale = best_model_data['length_scale']
        signal_var = best_model_data['signal_var']
        noise_var = best_model_data['noise_var']
        
        K_full = rbf_kernel(X_norm, X_norm, length_scale, signal_var) + noise_var * np.eye(n)
        
        try:
            w_full = solve(K_full, y_centered, assume_a='pos')
        except:
            w_full = np.linalg.lstsq(K_full, y_centered, rcond=None)[0]
            
        def predict(X):
            X_test_norm = (X - min_vals) / ranges
            K_test = rbf_kernel(X_test_norm, X_norm, length_scale, signal_var)
            return K_test @ w_full + y_mean

    return predict
