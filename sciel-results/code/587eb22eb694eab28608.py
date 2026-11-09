"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    t_start = time.time()
    deadline = 280.0  # safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Normalize features to [0, 1] range for better kernel behavior
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X - X_min) / X_range
    
    def compute_kernel(X1, X2, length_scales, poly_degree, poly_gamma, use_rbf=True, use_poly=True, use_linear=True, rbf_weight=1.0, poly_weight=1.0, lin_weight=1.0):
        """Compute composite kernel matrix."""
        # RBF kernel
        if use_rbf:
            # Squared distances
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            K_rbf = rbf_weight * np.exp(-0.5 * sq_dists / (length_scales ** 2))
        else:
            K_rbf = np.zeros((len(X1), len(X2)))
        
        # Polynomial kernel
        if use_poly:
            K_poly = poly_weight * (poly_gamma * X1 @ X2.T + 1) ** poly_degree
        else:
            K_poly = np.zeros((len(X1), len(X2)))
        
        # Linear kernel
        if use_linear:
            K_lin = lin_weight * (X1 @ X2.T)
        else:
            K_lin = np.zeros((len(X1), len(X2)))
        
        return K_rbf + K_poly + K_lin
    
    # Grid search over hyperparameters
    best_alpha = None
    best_score = np.inf
    best_params = None
    
    # Hyperparameter ranges
    length_scales = [0.1, 0.3, 0.5, 1.0, 2.0]
    poly_degrees = [2, 3, 4]
    poly_gammas = [0.1, 0.5, 1.0]
    ridge_alphas = [1e-4, 1e-3, 1e-2, 0.1, 1.0]
    rbf_weights = [0.5, 1.0, 2.0]
    poly_weights = [0.1, 0.5, 1.0]
    lin_weights = [0.0, 0.5, 1.0]
    
    # Use a subset of the grid to stay within time budget
    # Prioritize more important parameters
    n_evals = 0
    max_evals = 2000  # Adjust based on time
    
    for ls in length_scales:
        if time.time() > deadline - 30:
            break
        for pd in poly_degrees:
            if time.time() > deadline - 30:
                break
            for pg in poly_gammas:
                if time.time() > deadline - 30:
                    break
                for rw in rbf_weights:
                    if time.time() > deadline - 30:
                        break
                    for pw in poly_weights:
                        if time.time() > deadline - 30:
                            break
                        for lw in lin_weights:
                            if time.time() > deadline - 30:
                                break
                            for alpha in ridge_alphas:
                                if time.time() > deadline - 30:
                                    break
                                try:
                                    K = compute_kernel(X_norm, X_norm, ls, pd, pg, 
                                                       use_rbf=True, use_poly=True, use_linear=True,
                                                       rbf_weight=rw, poly_weight=pw, lin_weight=lw)
                                    # Ridge regression: solve (K + alpha*I) alpha_vec = y
                                    I = np.eye(n)
                                    A = K + alpha * I
                                    alpha_vec = np.linalg.solve(A, y)
                                    
                                    # Cross-validation-like score: use training MSE as proxy
                                    # Better: use a simple holdout or just trust the fit
                                    pred_train = K @ alpha_vec
                                    mse = np.mean((pred_train - y) ** 2)
                                    var_y = np.var(y)
                                    if var_y > 0:
                                        score = mse / var_y
                                    else:
                                        score = mse
                                    
                                    if score < best_score:
                                        best_score = score
                                        best_alpha = alpha_vec
                                        best_params = (ls, pd, pg, rw, pw, lw, alpha)
                                    n_evals += 1
                                except:
                                    continue
    
    # If no valid params found, use a default
    if best_alpha is None:
        # Fallback: simple RBF kernel with reasonable defaults
        ls = 0.5
        K = compute_kernel(X_norm, X_norm, ls, 2, 1.0, use_rbf=True, use_poly=False, use_linear=False)
        alpha = 0.1
        A = K + alpha * np.eye(n)
        best_alpha = np.linalg.solve(A, y)
        best_params = (ls, 2, 1.0, 1.0, 0.0, 0.0, alpha)
    
    ls, pd, pg, rw, pw, lw, alpha = best_params
    
    # Precompute kernel between training points for fast prediction
    K_train = compute_kernel(X_norm, X_norm, ls, pd, pg, 
                             use_rbf=True, use_poly=True, use_linear=True,
                             rbf_weight=rw, poly_weight=pw, lin_weight=lw)
    
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_norm = (X_test - X_min) / X_range
        K_test = compute_kernel(X_test_norm, X_norm, ls, pd, pg, 
                                use_rbf=True, use_poly=True, use_linear=True,
                                rbf_weight=rw, poly_weight=pw, lin_weight=lw)
        return K_test @ best_alpha
    
    return predict
# EVOLVE-BLOCK-END
