"""
Strategy: Gaussian Process Regression with Adaptive Length Scale.
Uses an RBF kernel. The length scale is optimized via grid search 
on log-spaced values using the log-marginal-likelihood.
This allows the model to adapt to the smoothness of the hidden function.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    """
    Fit a Gaussian Process model to the training data.
    Returns a predict function.
    """
    start_time = time.time()
    
    # Ensure X is 2D
    X = np.atleast_2d(X_train)
    if X.shape[0] == 1:
        X = X.T
    y = np.asarray(y_train)
    
    n, d = X.shape
    
    # Standardize X to improve numerical stability of the kernel
    # and allow length scale to be on a comparable scale
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    # Define the RBF kernel
    def rbf_kernel(X1, X2, length_scale):
        # Squared Euclidean distance
        # (x1 - x2)^2 = |x1|^2 - 2 x1.x2 + |x2|^2
        sq_dists = np.sum(X1**2, axis=1)[:, None] - 2 * X1 @ X2.T + np.sum(X2**2, axis=1)[None, :]
        sq_dists = np.maximum(sq_dists, 0) # Numerical stability
        return np.exp(-0.5 * sq_dists / (length_scale ** 2))
    
    # Noise level: assume small noise since data is noiseless, but add small epsilon for numerical stability
    noise_var = 1e-6
    
    # Search for optimal length scale
    # Log-spaced search over a wide range
    # The range of X_norm is roughly [-2, 2] or similar.
    # Length scale in [0.05, 10]
    ls_candidates = np.logspace(-1, 1, 50)
    
    best_ls = 1.0
    best_log_lik = -np.inf
    
    # Pre-compute X_norm @ X_norm.T? No, we need kernel matrices for different ls.
    # However, we can compute the pairwise distances once if we scale appropriately, 
    # but RBF depends on ls. 
    # Let's just compute K for each candidate.
    
    # Optimization loop
    for ls in ls_candidates:
        if time.time() - start_time > 280:
            break
            
        K = rbf_kernel(X_norm, X_norm, ls)
        # Add noise to diagonal
        K[np.diag_indices_from(K)] += noise_var
        
        try:
            # Cholesky decomposition for stability and likelihood calculation
            L = np.linalg.cholesky(K)
            
            # Log-likelihood: -0.5 * (y^T K^-1 y + log(det(K)) + n * log(2*pi))
            # log(det(K)) = 2 * sum(log(diag(L)))
            # K^-1 y = solve(L^T, solve(L, y))
            
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            log_det = 2 * np.sum(np.log(np.diag(L)))
            
            log_lik = -0.5 * (y @ alpha + log_det + n * np.log(2 * np.pi))
            
            if log_lik > best_log_lik:
                best_log_lik = log_lik
                best_ls = ls
                # Store Cholesky factor for prediction if this is the best so far?
                # We can re-compute it later, or store it. Storing is faster if we hit time limit.
                best_L = L
                best_alpha = alpha
                
        except np.linalg.LinAlgError:
            continue
            
    if time.time() - start_time > 290:
        # Fallback if optimization took too long: use median ls or default
        best_ls = 1.0
        K = rbf_kernel(X_norm, X_norm, best_ls)
        K[np.diag_indices_from(K)] += noise_var
        try:
            best_L = np.linalg.cholesky(K)
            best_alpha = np.linalg.solve(best_L.T, np.linalg.solve(best_L, y))
        except:
            # Last resort: simple linear regression fallback
            # (Should not happen usually)
            w = np.zeros(d+1)
            X_aug = np.column_stack([np.ones(n), X])
            w, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
            def predict_fallback(X_pred):
                X_p = np.atleast_2d(X_pred)
                if X_p.shape[0] == 1:
                    X_p = X_p.T
                X_p_norm = (X_p - X_mean) / X_std
                X_p_aug = np.column_stack([np.ones(len(X_p)), X_p_norm])
                return X_p_aug @ w
            return predict_fallback

    def predict(X_pred):
        X_p = np.atleast_2d(X_pred)
        if X_p.shape[0] == 1:
            X_p = X_p.T
        X_p_norm = (X_p - X_mean) / X_std
        
        # Compute kernel between test and train
        K_star = rbf_kernel(X_p_norm, X_norm, best_ls)
        
        # Prediction: K_star @ K_train^-1 @ y
        # We have L such that K = L L^T
        # K^-1 y = L^-T L^-1 y = alpha
        # So prediction is K_star @ alpha
        preds = K_star @ best_alpha
        
        return preds

    return predict
# EVOLVE-BLOCK-END
