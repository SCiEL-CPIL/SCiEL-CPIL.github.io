"""
Symbolic Regression via Gaussian Process with Matérn-5/2 Kernel and MLE.

Strategy:
1. Normalize inputs to [0, 1] to stabilize kernel computations.
2. Use a Matérn-5/2 kernel with anisotropic length scales (one per dimension).
   This allows the model to adapt to different input sensitivities, which is
   critical for multi-dimensional equations where variables contribute differently.
3. Optimize hyperparameters (log-length-scales, log-signal-variance) using
   exact Maximum Likelihood Estimation (MLE).
   - Since the data is noiseless, we set observation noise sigma^2 to a small
     epsilon (1e-8) to ensure numerical stability and near-perfect interpolation.
   - The MLE process balances fit quality (likelihood of y given X, theta) and
     model complexity (prior on f), automatically selecting the appropriate
     smoothness (length scales) for each dimension.
4. Use multi-start optimization to avoid local minima in the hyperparameter space.
5. Return a predict function that computes the GP posterior mean.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    
    n_samples, n_features = X.shape
    
    # 1. Normalize inputs to [0, 1]
    min_vals = X.min(axis=0, keepdims=True)
    max_vals = X.max(axis=0, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    
    X_norm = (X - min_vals) / range_vals
    
    # 2. Define Matérn-5/2 kernel
    # K(x, x') = sigma^2 * (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)
    # where r = sqrt(sum((x_i - x'_i)^2 / l_i^2))
    
    # Hyperparameters to optimize:
    # h = [log(l_1), ..., log(l_d), log(sigma)]
    # l_i > 0, sigma > 0
    
    def log_likelihood(params, X, y, is_neg=False):
        """
        Computes the negative log marginal likelihood.
        params: [log_l1, ..., log_ld, log_sigma]
        """
        try:
            log_ls = params[:n_features]
            log_sigma = params[n_features]
            
            ls = np.exp(log_ls)
            sigma = np.exp(log_sigma)
            
            # Ensure length scales are positive and not too small/large
            # Clip to reasonable range to prevent numerical issues
            ls = np.clip(ls, 1e-3, 10.0)
            sigma = np.clip(sigma, 1e-3, 100.0)
            
            # Compute scaled distances
            # X_scaled = X / ls
            X_scaled = X / ls
            
            # Compute squared distances
            # ||x - x'||^2 = ||x||^2 + ||x'||^2 - 2 x . x'
            X_sq = np.sum(X_scaled**2, axis=1, keepdims=True)
            sq_dists = X_sq + X_sq.T - 2 * (X_scaled @ X_scaled.T)
            sq_dists = np.maximum(sq_dists, 0)
            
            r = np.sqrt(sq_dists)
            
            # Matérn-5/2 kernel
            sqrt5 = np.sqrt(5.0)
            K = sigma**2 * (1 + sqrt5*r + (5/3)*r**2) * np.exp(-sqrt5*r)
            
            # Add noise for stability
            noise = 1e-8
            K += noise * np.eye(n_samples)
            
            # Cholesky decomposition for stable determinant and solve
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # If Cholesky fails, add more jitter and retry once
                K += 1e-6 * np.eye(n_samples)
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    return 1e10 if not is_neg else -1e10
            
            # Solve K alpha = y
            # alpha = L^{-T} L^{-1} y
            Ly = np.linalg.solve(L, y)
            alpha = np.linalg.solve(L.T, Ly)
            
            # Log marginal likelihood
            # -0.5 * (y^T alpha + log(det(K)) + n log(2*pi))
            log_det_K = 2.0 * np.sum(np.log(np.diag(L)))
            
            nll = 0.5 * (np.dot(y, alpha) + log_det_K + n_samples * np.log(2.0 * np.pi))
            
            if is_neg:
                return -nll
            else:
                return nll
                
        except Exception:
            return 1e10 if not is_neg else -1e10

    # 3. Multi-start optimization for hyperparameters
    # Initial guesses for length scales:
    # Try a few different starting points for the length scales.
    # Since data is normalized to [0,1], typical scales are 0.1 to 1.0.
    # log(0.1) = -2.3, log(1.0) = 0.
    
    best_nll = float('inf')
    best_params = None
    
    # Generate starting points
    start_points = []
    
    # 1. Isotropic starts
    for log_l_start in [-2.0, -1.0, 0.0, 1.0]:
        start_points.append(np.full(n_features + 1, log_l_start))
        
    # 2. Random starts
    np.random.seed(42)
    for _ in range(10):
        log_ls = np.random.uniform(-3.0, 1.0, size=n_features)
        log_sigma = np.random.uniform(-1.0, 2.0)
        start_points.append(np.concatenate([log_ls, [log_sigma]]))
        
    # 3. Anisotropic starts based on feature variance
    # If a feature has low variance in y correlation (hard to estimate), 
    # maybe it's less important? 
    # Simple heuristic: start with different scales for each dim
    for i in range(n_features):
        # Try emphasizing one dimension
        log_ls = np.zeros(n_features)
        log_ls[i] = -2.0 # Small scale -> high frequency
        log_sigma = 0.0
        start_points.append(np.concatenate([log_ls, [log_sigma]]))
        
    # Optimize
    for sp in start_points:
        if time.time() > deadline - 20:
            break
            
        try:
            res = minimize(
                lambda p: log_likelihood(p, X_norm, y, is_neg=True),
                sp,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if res.success or res.fun < best_nll:
                # Check if the result is valid
                val = log_likelihood(res.x, X_norm, y, is_neg=True)
                if val < best_nll:
                    best_nll = val
                    best_params = res.x
        except Exception:
            continue
            
    # If optimization failed completely, fallback to a simple isotropic RBF-like GP
    if best_params is None:
        # Use a default Matérn-5/2 with l=0.5, sigma=1
        best_params = np.concatenate([np.full(n_features, np.log(0.5)), [0.0]])

    # 4. Extract final hyperparameters
    log_ls = best_params[:n_features]
    log_sigma = best_params[n_features]
    
    ls = np.exp(log_ls)
    sigma = np.exp(log_sigma)
    
    # Ensure stability
    ls = np.clip(ls, 1e-3, 10.0)
    sigma = np.clip(sigma, 1e-3, 100.0)
    
    # 5. Fit the final GP model
    X_scaled = X_norm / ls
    X_sq = np.sum(X_scaled**2, axis=1, keepdims=True)
    sq_dists = X_sq + X_sq.T - 2 * (X_scaled @ X_scaled.T)
    sq_dists = np.maximum(sq_dists, 0)
    r = np.sqrt(sq_dists)
    
    sqrt5 = np.sqrt(5.0)
    K = sigma**2 * (1 + sqrt5*r + (5/3)*r**2) * np.exp(-sqrt5*r)
    
    noise = 1e-8
    K += noise * np.eye(n_samples)
    
    # Solve for alpha
    try:
        L = np.linalg.cholesky(K)
        Ly = np.linalg.solve(L, y)
        alpha = np.linalg.solve(L.T, Ly)
    except np.linalg.LinAlgError:
        # Fallback to least squares if Cholesky fails
        try:
            alpha = np.linalg.lstsq(K, y, rcond=None)[0]
        except Exception:
            # Final fallback: polynomial
            def features(X):
                cols = [np.ones(len(X))] + [X[:, j] for j in range(X.shape[1])] + [X[:, i] * X[:, j] for i in range(X.shape[1]) for j in range(i, X.shape[1])]
                return np.column_stack(cols)
            w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
            return lambda X: features(X) @ w

    # 6. Define predict function
    def predict(X):
        X = np.asarray(X, dtype=float)
        X_norm_test = (X - min_vals) / range_vals
        
        X_scaled_test = X_norm_test / ls
        X_scaled_train = X_norm / ls
        
        # Compute distances between test and train
        # ||x_test - x_train||^2
        X_test_sq = np.sum(X_scaled_test**2, axis=1, keepdims=True)
        X_train_sq = np.sum(X_scaled_train**2, axis=1, keepdims=True)
        sq_dists_test = X_test_sq + X_train_sq.T - 2 * (X_scaled_test @ X_scaled_train.T)
        sq_dists_test = np.maximum(sq_dists_test, 0)
        
        r_test = np.sqrt(sq_dists_test)
        
        # Kernel between test and train
        K_test = sigma**2 * (1 + sqrt5*r_test + (5/3)*r_test**2) * np.exp(-sqrt5*r_test)
        
        # Prediction
        y_pred = K_test @ alpha
        
        return y_pred

    return predict
# EVOLVE-BLOCK-END
