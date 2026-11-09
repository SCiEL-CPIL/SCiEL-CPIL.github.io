"""Kernel ridge regression with composite kernel for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import cho_factor, cho_solve

def fit(X_train, y_train):
    n, d = X_train.shape
    
    # Normalize inputs to [0, 1] range
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    padding = 0.1 * (X_max - X_min + 1e-8)
    X_min = X_min - padding
    X_max = X_max + padding
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X_train - X_min) / X_range
    
    # Precompute for efficient kernel evaluation
    # We'll use a composite kernel: RBF + Polynomial
    
    def compute_kernel(X1, X2, length_scale=1.0, poly_degree=3, poly_gamma=1.0):
        """Compute composite kernel: RBF + Polynomial."""
        
        # 1. RBF Kernel
        sq_dists = (np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        K_rbf = np.exp(-sq_dists / (2 * length_scale**2))
        
        # 2. Polynomial Kernel
        K_poly = (poly_gamma * (X1 @ X2.T) + 1.0) ** poly_degree
        
        # Combine: RBF + Polynomial with equal weight
        K_total = K_rbf + K_poly
        
        return K_total
    
    # Hyperparameter grid
    # Keep grid small to ensure fast execution
    length_scales = [0.05, 0.1, 0.2, 0.5, 1.0]
    poly_degrees = [2, 3, 4]
    poly_gammas = [0.5, 1.0, 2.0]
    
    # Small regularization for numerical stability
    lambda_reg = 1e-8
    
    best_cv_score = np.inf
    best_params = {'length_scale': 0.5, 'poly_degree': 3, 'poly_gamma': 1.0}
    best_alpha = None
    
    for ls in length_scales:
        for pd in poly_degrees:
            for pg in poly_gammas:
                K_raw = compute_kernel(X_norm, X_norm, ls, pd, pg)
                K_reg = K_raw + lambda_reg * np.eye(n)
                
                try:
                    # Use Cholesky for stability
                    L, lower = cho_factor(K_reg, lower=lower)
                    alpha = cho_solve((L, lower), y_train)
                    
                    # LOOCV score using hat matrix diagonal
                    # H = K_raw @ (K_raw + lambda I)^{-1}
                    # diag_H = diag(K_raw @ inv(K_reg))
                    # Efficient: solve K_reg X = K_raw, then diag(X)
                    # Or use: h_ii = 1 - lambda * (K_reg^{-1})_{ii}
                    # Actually for KRR: h_ii = sum_j K_ij alpha_j / y_i is not right.
                    # Correct: y_hat_loo_i = y_i - (y_i - y_hat_i) / (1 - h_ii)
                    # where h_ii = (K (K + lambda I)^{-1})_{ii}
                    
                    # Compute h_ii efficiently
                    # Solve K_reg Z = K_raw, then h_ii = Z_ii
                    Z = cho_solve((L, lower), K_raw)
                    diag_H = np.diag(Z)
                    
                    y_hat = K_raw @ alpha
                    denom = 1.0 - diag_H
                    denom_clamped = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
                    y_hat_loo = y_train - (y_train - y_hat) / denom_clamped
                    cv_mse = np.mean((y_hat_loo - y_train)**2)
                    
                    if cv_mse < best_cv_score:
                        best_cv_score = cv_mse
                        best_params = {'length_scale': ls, 'poly_degree': pd, 'poly_gamma': pg}
                        best_alpha = alpha
                except:
                    continue
    
    # Final fit with best parameters
    K_raw_final = compute_kernel(X_norm, X_norm, 
                                 best_params['length_scale'], 
                                 best_params['poly_degree'], 
                                 best_params['poly_gamma'])
    K_reg_final = K_raw_final + lambda_reg * np.eye(n)
    L_final, lower_final = cho_factor(K_reg_final, lower=lower_final)
    alpha_final = cho_solve((L_final, lower_final), y_train)
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_norm_test = (X - X_min) / X_range
        K_test = compute_kernel(X_norm_test, X_norm, 
                                best_params['length_scale'], 
                                best_params['poly_degree'], 
                                best_params['poly_gamma'])
        return K_test @ alpha_final
    
    return predict
# EVOLVE-BLOCK-END
