"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def fit(X_train, y_train):
    t_start = time.time()
    deadline = 280.0  # safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Normalize features to [0, 1] range
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X - X_min) / X_range
    
    def rbf_kernel(X1, X2, length_scale):
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
        sq_dists = np.maximum(sq_dists, 0)
        return np.exp(-0.5 * sq_dists / (length_scale ** 2))
    
    def poly_kernel(X1, X2, degree, gamma):
        return (gamma * X1 @ X2.T + 1) ** degree
    
    def periodic_kernel(X1, X2, period):
        # Periodic kernel: K(x, x') = exp(-2 * sum(sin^2(pi * (x - x') / period)))
        # For simplicity, use a single period across all dimensions
        diff = X1[:, None, :] - X2[None, :, :]
        return np.exp(-2.0 * np.sum(np.sin(np.pi * diff / period) ** 2, axis=2))
    
    def compute_kernel(X1, X2, params):
        """Composite kernel with RBF, polynomial, and periodic components."""
        K = np.zeros((len(X1), len(X2)))
        
        if params['use_rbf'] and params['rbf_weight'] > 0:
            K += params['rbf_weight'] * rbf_kernel(X1, X2, params['rbf_ls'])
        
        if params['use_poly'] and params['poly_weight'] > 0:
            K += params['poly_weight'] * poly_kernel(X1, X2, params['poly_deg'], params['poly_gamma'])
        
        if params['use_periodic'] and params['periodic_weight'] > 0:
            K += params['periodic_weight'] * periodic_kernel(X1, X2, params['period'])
        
        return K
    
    def loocv_score(X, y, params, alpha):
        """Compute LOOCV normalized MSE for kernel ridge regression."""
        K = compute_kernel(X, X, params)
        I = np.eye(len(X))
        A = K + alpha * I
        
        try:
            # For LOOCV: prediction for i is (K @ alpha_vec)_i - K[i,i] * alpha_vec[i] / (1 + alpha)
            # But more efficiently: use the formula with inverse
            # Actually, for kernel ridge: alpha_vec = (K + alpha*I)^{-1} y
            # LOOCV prediction: y_hat_i = (K @ alpha_vec)_i - K[i,i] * alpha_vec[i] / (1 + alpha)
            # This is not quite right. Let me use the proper formula.
            
            # Proper LOOCV for kernel ridge:
            # h_ii = (K + alpha*I)^{-1}_{ii}
            # y_hat_i = y_i - (K @ alpha_vec)_i * h_ii / (1 + alpha * h_ii)
            # Actually, the standard formula is:
            # y_hat_i = y_i - (K @ alpha_vec)_i * h_ii / (1 + alpha)
            # where h_ii is the i-th diagonal of (K + alpha*I)^{-1}
            
            # Let me just compute the full prediction and use a simple holdout-like approach
            # Or better: compute the leave-one-out predictions directly
            
            alpha_vec = np.linalg.solve(A, y)
            pred = K @ alpha_vec
            
            # LOOCV formula for kernel ridge regression:
            # y_hat_i = y_i - (K @ alpha_vec)_i * H_ii / (1 + alpha)
            # where H = (K + alpha*I)^{-1}
            H = np.linalg.inv(A)
            H_diag = np.diag(H)
            
            y_hat = y - pred * H_diag / (1 + alpha)
            
            mse = np.mean((y_hat - y) ** 2)
            var_y = np.var(y)
            if var_y > 0:
                return mse / var_y
            else:
                return mse
        except:
            return np.inf
    
    # Hyperparameter search with LOOCV
    best_score = np.inf
    best_params = None
    best_alpha = None
    
    # Define search space
    rbf_ls_values = [0.1, 0.3, 0.5, 1.0, 2.0]
    poly_deg_values = [2, 3, 4]
    poly_gamma_values = [0.1, 0.5, 1.0]
    period_values = [1.0, 2.0, 3.0, 5.0]
    rbf_weights = [0.0, 0.5, 1.0]
    poly_weights = [0.0, 0.5, 1.0]
    periodic_weights = [0.0, 0.5, 1.0]
    ridge_alphas = [1e-4, 1e-3, 1e-2, 0.1]
    
    # Create a reduced search space to fit within time budget
    # Prioritize combinations that are likely to work well
    search_configs = []
    
    # RBF only
    for ls in rbf_ls_values:
        for rw in [0.5, 1.0]:
            for alpha in ridge_alphas:
                search_configs.append({
                    'use_rbf': True, 'rbf_ls': ls, 'rbf_weight': rw,
                    'use_poly': False, 'poly_deg': 0, 'poly_gamma': 1.0, 'poly_weight': 0.0,
                    'use_periodic': False, 'period': 1.0, 'periodic_weight': 0.0,
                    'alpha': alpha
                })
    
    # Polynomial only
    for deg in poly_deg_values:
        for pg in poly_gamma_values:
            for pw in [0.5, 1.0]:
                for alpha in ridge_alphas:
                    search_configs.append({
                        'use_rbf': False, 'rbf_ls': 1.0, 'rbf_weight': 0.0,
                        'use_poly': True, 'poly_deg': deg, 'poly_gamma': pg, 'poly_weight': pw,
                        'use_periodic': False, 'period': 1.0, 'periodic_weight': 0.0,
                        'alpha': alpha
                    })
    
    # RBF + Polynomial
    for ls in [0.3, 0.5, 1.0]:
        for deg in [2, 3]:
            for pg in [0.5, 1.0]:
                for rw in [0.5, 1.0]:
                    for pw in [0.5, 1.0]:
                        for alpha in [1e-3, 1e-2]:
                            search_configs.append({
                                'use_rbf': True, 'rbf_ls': ls, 'rbf_weight': rw,
                                'use_poly': True, 'poly_deg': deg, 'poly_gamma': pg, 'poly_weight': pw,
                                'use_periodic': False, 'period': 1.0, 'periodic_weight': 0.0,
                                'alpha': alpha
                            })
    
    # Periodic only
    for period in period_values:
        for pweight in [0.5, 1.0]:
            for alpha in ridge_alphas:
                search_configs.append({
                    'use_rbf': False, 'rbf_ls': 1.0, 'rbf_weight': 0.0,
                    'use_poly': False, 'poly_deg': 0, 'poly_gamma': 1.0, 'poly_weight': 0.0,
                    'use_periodic': True, 'period': period, 'periodic_weight': pweight,
                    'alpha': alpha
                })
    
    # RBF + Periodic
    for ls in [0.3, 0.5, 1.0]:
        for period in [1.0, 2.0, 3.0]:
            for rw in [0.5, 1.0]:
                for pweight in [0.5, 1.0]:
                    for alpha in [1e-3, 1e-2]:
                        search_configs.append({
                            'use_rbf': True, 'rbf_ls': ls, 'rbf_weight': rw,
                            'use_poly': False, 'poly_deg': 0, 'poly_gamma': 1.0, 'poly_weight': 0.0,
                            'use_periodic': True, 'period': period, 'periodic_weight': pweight,
                            'alpha': alpha
                        })
    
    # Evaluate each configuration with LOOCV
    n_evals = 0
    for params in search_configs:
        if time.time() > deadline - 30:
            break
        
        alpha = params['alpha']
        try:
            score = loocv_score(X_norm, y, params, alpha)
            
            if score < best_score:
                best_score = score
                best_params = params.copy()
                best_alpha = alpha
                # Store the alpha vector for the best params
                K = compute_kernel(X_norm, X_norm, best_params)
                A = K + best_alpha * np.eye(n)
                best_alpha_vec = np.linalg.solve(A, y)
            n_evals += 1
        except:
            continue
    
    # If no valid params found, use a default
    if best_params is None:
        best_params = {
            'use_rbf': True, 'rbf_ls': 0.5, 'rbf_weight': 1.0,
            'use_poly': False, 'poly_deg': 0, 'poly_gamma': 1.0, 'poly_weight': 0.0,
            'use_periodic': False, 'period': 1.0, 'periodic_weight': 0.0,
            'alpha': 0.01
        }
        best_alpha = 0.01
        K = compute_kernel(X_norm, X_norm, best_params)
        A = K + best_alpha * np.eye(n)
        best_alpha_vec = np.linalg.solve(A, y)
    
    # Precompute for fast prediction
    K_train = compute_kernel(X_norm, X_norm, best_params)
    
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_norm = (X_test - X_min) / X_range
        K_test = compute_kernel(X_test_norm, X_norm, best_params)
        return K_test @ best_alpha_vec
    
    return predict
# EVOLVE-BLOCK-END
