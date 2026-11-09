# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 315.0  # 15s margin for 330s kill
    
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    
    # Handle constant or near-constant targets
    if y_var < 1e-10:
        return lambda X: np.full(X.shape[0], y_mean)
    
    def check_time():
        return time.time() < deadline

    # Normalize inputs to [0, 1] to stabilize kernel computations
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    
    X_norm = (X_train - X_min) / X_range
    y_norm = (y_train - y_mean) / np.sqrt(y_var)
    
    # Matérn 5/2 kernel
    def matern_52_kernel(X1, X2, theta):
        # theta: [log_length_scales (d), log_signal_var, log_noise]
        # For simplicity, we might use a single length scale or separate ones.
        # Let's try separate length scales for better adaptability, but it increases params.
        # Given d <= 3, separate is fine.
        
        log_ls = theta[:d]
        log_sigma_f = theta[d]
        log_sigma_n = theta[d+1]
        
        ls = np.exp(log_ls)
        sigma_f = np.exp(log_sigma_f)
        sigma_n = np.exp(log_sigma_n)
        
        # Squared distances
        # cdist returns pairwise Euclidean distances
        D = cdist(X1, X2, metric='euclidean')
        
        # Scale distances by length scales
        # D_scaled = D / ls
        # We need to handle broadcasting carefully
        # D is (n1, n2). ls is (d,).
        # We want to compute sum((x1_i - x2_j)^2 / l_k^2)
        
        # Efficient way:
        # X1_scaled = X1 / ls
        # X2_scaled = X2 / ls
        # D_scaled = cdist(X1_scaled, X2_scaled)
        
        X1_scaled = X1 / ls
        X2_scaled = X2 / ls
        D_scaled = cdist(X1_scaled, X2_scaled, metric='euclidean')
        
        # K = sigma_f^2 * (1 + sqrt(5)*D_scaled + 5/3*D_scaled^2) * exp(-sqrt(5)*D_scaled)
        sqrt5 = np.sqrt(5.0)
        term1 = 1.0 + sqrt5 * D_scaled + (5.0 / 3.0) * D_scaled**2
        term2 = np.exp(-sqrt5 * D_scaled)
        
        K = (sigma_f**2) * term1 * term2
        
        # Add noise
        if X1 is X2: # Only add noise for training matrix
            K[np.diag_indices_from(K)] += sigma_n**2
            
        return K

    def log_likelihood(theta):
        try:
            K = matern_52_kernel(X_norm, X_norm, theta)
            
            # Solve K alpha = y
            # Use Cholesky for stability
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Add small jitter
                K += 1e-6 * np.eye(n)
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    return -1e100 # Very poor likelihood
            
            # alpha = L^-T L^-1 y
            v = np.linalg.solve(L, y_norm)
            alpha = np.linalg.solve(L.T, v)
            
            # log p(y|X) = -0.5 * y^T K^-1 y - 0.5 * log det(K) - n/2 * log(2*pi)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            ll = -0.5 * (y_norm @ alpha) - 0.5 * log_det - 0.5 * n * np.log(2.0 * np.pi)
            
            if np.isnan(ll) or np.isinf(ll):
                return -1e100
                
            return ll
        except Exception:
            return -1e100

    def negative_log_likelihood(theta):
        return -log_likelihood(theta)

    # Initial guesses
    # Length scales: assume inputs are in [0,1], so length scale ~ 0.5
    init_ls = np.log(np.ones(d) * 0.5)
    init_sigma_f = np.log(1.0)
    init_sigma_n = np.log(1e-2)
    
    theta0 = np.concatenate([init_ls, [init_sigma_f, init_sigma_n]])
    
    # Bounds: length scales in [0.01, 10], sigma_f in [0.01, 100], sigma_n in [1e-5, 10]
    bounds_ls = [(np.log(0.01), np.log(10.0))] * d
    bounds_sf = [(np.log(0.01), np.log(100.0))]
    bounds_sn = [(np.log(1e-5), np.log(10.0))]
    bounds = bounds_ls + bounds_sf + bounds_sn

    best_theta = theta0
    best_ll = negative_log_likelihood(theta0)
    
    # Optimization
    # Use L-BFGS-B
    try:
        res = minimize(negative_log_likelihood, theta0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
        
        if res.success or res.fun < best_ll:
            best_theta = res.x
            best_ll = res.fun
    except Exception:
        pass
    
    # If optimization failed or took too long, we might try a simpler grid search or just use init
    # But L-BFGS-B is usually fast for n=200, d<=3.
    
    # Build the model
    def predict(X_raw):
        X_test = (X_raw - X_min) / X_range
        K_test = matern_52_kernel(X_test, X_norm, best_theta)
        
        # We need K_train_inv
        # Re-compute K_train with best_theta
        K_train = matern_52_kernel(X_norm, X_norm, best_theta)
        
        try:
            L = np.linalg.cholesky(K_train)
        except np.linalg.LinAlgError:
            K_train += 1e-6 * np.eye(n)
            L = np.linalg.cholesky(K_train)
            
        v = np.linalg.solve(L, y_norm)
        alpha = np.linalg.solve(L.T, v)
        
        y_pred_norm = K_test @ alpha
        y_pred = y_pred_norm * np.sqrt(y_var) + y_mean
        
        # Ensure finite
        y_pred = np.where(np.isfinite(y_pred), y_pred, y_mean)
        
        return y_pred

    # Sanity check
    try:
        y_pred_check = predict(X_train)
        if not np.all(np.isfinite(y_pred_check)):
            raise ValueError("Non-finite predictions")
        mse_check = np.mean((y_pred_check - y_train)**2)
        if mse_check > 10 * y_var:
            # Fallback to mean if GP is terrible
            return lambda X: np.full(X.shape[0], y_mean)
    except Exception:
        return lambda X: np.full(X.shape[0], y_mean)

    return predict

# EVOLVE-BLOCK-END
