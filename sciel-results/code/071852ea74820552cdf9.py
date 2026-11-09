import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# EVOLVE-BLOCK-START

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 300.0  # 300s limit, leaving margin

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)

    # Handle potential 1D inputs by reshaping to 2D
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)

    n_train, d = X_train.shape

    # Standardize inputs and outputs for better numerical conditioning
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std == 0] = 1.0
    
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std == 0:
        y_std = 1.0

    X_train_std = (X_train - X_mean) / X_std
    y_train_std = (y_train - y_mean) / y_std

    # Precompute pairwise distances for kernel evaluation
    # Matern 5/2 kernel: k(x, x') = sigma^2 * (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)
    # where r = ||x - x'|| / l
    
    # We optimize log(l) and log(sigma) to ensure positivity
    # Fixed noise variance for stability
    noise_var = 1e-6

    def log_marginal_likelihood(params):
        log_l, log_sigma = params
        l = np.exp(log_l)
        sigma = np.exp(log_sigma)
        
        # Calculate distances
        # Using cdist for efficiency
        dists = cdist(X_train_std, X_train_std, metric='euclidean')
        r = dists / l
        
        # Kernel matrix
        sqrt_5_r = np.sqrt(5.0) * r
        K = (sigma**2) * (1.0 + sqrt_5_r + (5.0/3.0) * r**2) * np.exp(-sqrt_5_r)
        
        # Add noise to diagonal
        K[np.diag_indices_from(K)] += noise_var

        try:
            # Cholesky decomposition
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # If Cholesky fails, return a large negative likelihood
            return -1e10

        # Solve L alpha = y
        alpha = np.linalg.solve(L, y_train_std.ravel())
        # Solve L^T v = alpha
        v = np.linalg.solve(L.T, alpha)
        
        # Log marginal likelihood
        # -0.5 * y^T K^-1 y - 0.5 * log(det(K)) - n/2 * log(2*pi)
        # det(K) = prod(diag(L))^2 = exp(2 * sum(log(diag(L))))
        # So 0.5 * log(det(K)) = sum(log(diag(L)))
        
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        mll = -0.5 * np.dot(y_train_std.ravel(), v) - 0.5 * log_det - 0.5 * n_train * np.log(2.0 * np.pi)
        
        return -mll # Minimize negative log likelihood

    # Initial guess for hyperparameters
    # Length scale typically around 1 in standardized space
    # Signal variance typically around 1
    x0 = np.array([0.0, 0.0]) # log(1), log(1)

    best_mll = np.inf
    best_params = x0

    # Optimize hyperparameters
    # Use a simple multi-start or single run if time is tight
    # Single run is usually sufficient for 1D
    try:
        res = minimize(log_marginal_likelihood, x0, method='L-BFGS-B', 
                       options={'maxiter': 100, 'ftol': 1e-6})
        if res.fun < best_mll:
            best_mll = res.fun
            best_params = res.x
    except Exception:
        pass

    # Check time
    if time.time() > deadline - 10.0:
        # Fallback to a simple linear fit if GP failed or took too long
        # But we should have a valid GP model from the initial guess if optimize failed
        pass

    l_opt = np.exp(best_params[0])
    sigma_opt = np.exp(best_params[1])

    # Precompute for prediction
    # We need to solve K + sigma_n^2 I for the training data to get weights
    # Or just compute the kernel matrix again
    
    dists_train = cdist(X_train_std, X_train_std, metric='euclidean')
    r_train = dists_train / l_opt
    sqrt_5_r_train = np.sqrt(5.0) * r_train
    K_train = (sigma_opt**2) * (1.0 + sqrt_5_r_train + (5.0/3.0) * r_train**2) * np.exp(-sqrt_5_r_train)
    K_train[np.diag_indices_from(K_train)] += noise_var

    try:
        L_train = np.linalg.cholesky(K_train)
        alpha_train = np.linalg.solve(L_train, y_train_std.ravel())
        v_train = np.linalg.solve(L_train.T, alpha_train)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if Cholesky fails
        alpha_train = np.linalg.pinv(K_train) @ y_train_std.ravel()
        v_train = np.linalg.pinv(K_train) @ y_train_std.ravel()

    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_std = (X - X_mean) / X_std_global
        
        dists_test = cdist(X_std, X_train_std, metric='euclidean')
        r_test = dists_test / l_opt
        sqrt_5_r_test = np.sqrt(5.0) * r_test
        K_test = (sigma_opt**2) * (1.0 + sqrt_5_r_test + (5.0/3.0) * r_test**2) * np.exp(-sqrt_5_r_test)
        
        # Prediction mean
        # y_pred_std = K_test @ (K_train + noise I)^-1 y_train_std
        # We have v_train = (K_train + noise I)^-1 y_train_std
        y_pred_std = K_test @ v_train
        
        # Unstandardize
        y_pred = y_pred_std * y_std + y_mean
        
        return y_pred.ravel()

    # Store X_std_global in closure
    X_std_global = X_std
    
    return predict

# EVOLVE-BLOCK-END
