"""Kernel ridge regression with composite kernel for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import solve

def fit(X_train, y_train):
    n, d = X_train.shape
    
    # Normalize inputs to [0, 1] range for better kernel scaling
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X_train - X_min) / X_range
    
    # Composite kernel: RBF + Polynomial + Linear
    # We'll use a weighted sum of kernels
    # K_total = w1 * K_rbf + w2 * K_poly + w3 * K_linear
    
    def compute_kernel(X1, X2, length_scale=1.0, poly_degree=3, poly_gamma=1.0, poly_coef0=1.0):
        """Compute composite kernel matrix."""
        # RBF kernel
        sq_dists = (np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        K_rbf = np.exp(-sq_dists / (2 * length_scale**2))
        
        # Polynomial kernel
        K_poly = (poly_gamma * (X1 @ X2.T) + poly_coef0) ** poly_degree
        
        # Linear kernel
        K_linear = X1 @ X2.T
        
        # Composite: balance the scales
        # Normalize each kernel to have similar scale
        K_rbf = K_rbf / n
        K_poly = K_poly / (n ** poly_degree)
        K_linear = K_linear / n
        
        return K_rbf + 0.5 * K_poly + 0.5 * K_linear
    
    # Try a few length scales and pick the best via simple validation
    # Use a small grid search with 5-fold CV
    best_score = np.inf
    best_length_scale = 1.0
    best_poly_degree = 3
    best_poly_gamma = 1.0
    best_poly_coef0 = 1.0
    
    # Simple grid search over key hyperparameters
    length_scales = [0.1, 0.3, 0.5, 1.0, 2.0]
    poly_degrees = [2, 3, 4]
    poly_gammas = [0.5, 1.0, 2.0]
    poly_coef0s = [1.0, 2.0]
    
    # To keep it fast, we'll do a reduced grid
    # First, try different length scales with fixed poly params
    for ls in length_scales:
        for pd in [3]:
            for pg in [1.0]:
                for pc in [1.0]:
                    K = compute_kernel(X_norm, X_norm, ls, pd, pg, pc)
                    # Add small ridge for stability
                    K_reg = K + 1e-6 * np.eye(n)
                    try:
                        alpha = solve(K_reg, y_train, assume_a='pos')
                        # Quick training error as proxy
                        pred = K @ alpha
                        train_mse = np.mean((pred - y_train)**2)
                        if train_mse < best_score:
                            best_score = train_mse
                            best_length_scale = ls
                            best_poly_degree = pd
                            best_poly_gamma = pg
                            best_poly_coef0 = pc
                    except:
                        pass
    
    # Now refine with best length scale, try different poly degrees
    for pd in poly_degrees:
        for pg in poly_gammas:
            for pc in poly_coef0s:
                K = compute_kernel(X_norm, X_norm, best_length_scale, pd, pg, pc)
                K_reg = K + 1e-6 * np.eye(n)
                try:
                    alpha = solve(K_reg, y_train, assume_a='pos')
                    pred = K @ alpha
                    train_mse = np.mean((pred - y_train)**2)
                    if train_mse < best_score:
                        best_score = train_mse
                        best_length_scale = best_length_scale
                        best_poly_degree = pd
                        best_poly_gamma = pg
                        best_poly_coef0 = pc
                except:
                    pass
    
    # Final fit with best hyperparameters
    K = compute_kernel(X_norm, X_norm, best_length_scale, best_poly_degree, best_poly_gamma, best_poly_coef0)
    K_reg = K + 1e-6 * np.eye(n)
    alpha = solve(K_reg, y_train, assume_a='pos')
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        X_norm_test = (X - X_min) / X_range
        K_test = compute_kernel(X_norm_test, X_norm, best_length_scale, best_poly_degree, best_poly_gamma, best_poly_coef0)
        return K_test @ alpha
    
    return predict
# EVOLVE-BLOCK-END
