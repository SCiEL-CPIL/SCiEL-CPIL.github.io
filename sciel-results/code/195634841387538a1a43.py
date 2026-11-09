import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    start_time = time.time()
    timeout = 250  # Safety margin before 300s limit
    
    n_samples, n_features = X_train.shape
    
    # Helper to check time
    def time_check():
        if time.time() - start_time > timeout:
            return True
        return False

    # Normalize X and y for better conditioning
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X_train - X_mean) / X_std
    
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std < 1e-10:
        y_norm = y_train
        y_scale = 1.0
    else:
        y_norm = (y_train - y_mean) / y_std
        y_scale = y_std

    # Strategy: Gaussian Process Regression with RBF Kernel
    # We assume the function is smooth. The RBF kernel allows for local flexibility.
    # Since data is noiseless, we use a very small noise variance to allow exact interpolation
    # while the length scale controls the complexity (smoothness).
    
    def log_likelihood(params, X, y, noise_var):
        # params: [log_l, log_sigma_f]
        l = np.exp(params[0])
        sigma_f = np.exp(params[1])
        
        # Compute RBF kernel
        # K_ij = sigma_f^2 * exp(-0.5 * ||x_i - x_j||^2 / l^2)
        D2 = cdist(X, X, metric='sqeuclidean')
        K = (sigma_f ** 2) * np.exp(-0.5 * D2 / (l ** 2))
        
        # Add noise
        K += noise_var * np.eye(n_samples)
        
        try:
            # Cholesky decomposition
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -1e10
            
        # Solve L alpha = y
        alpha = np.linalg.solve(L, y)
        # Solve L^T v = alpha
        v = np.linalg.solve(L.T, alpha)
        
        # Log likelihood: -0.5 * y^T K^-1 y - 0.5 * log|K| - 0.5 * N * log(2*pi)
        # log|K| = 2 * sum(log(diag(L)))
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        
        ll = -0.5 * (y @ v + log_det) - 0.5 * n_samples * np.log(2 * np.pi)
        
        return ll

    # We will optimize log_l and log_sigma_f
    # Initial guesses:
    # l: typical distance between points. For normalized data, ~1.0.
    # sigma_f: std of y, which is 1.0.
    
    # Since data is noiseless, we set noise_var very small.
    # If noise_var is too small, K becomes singular. 
    # We can also optimize noise_var, but constraining it is safer for "exact" fit logic.
    # Let's try a few fixed small noise values and pick the best LL, 
    # or optimize it with a lower bound.
    
    best_ll = -np.inf
    best_params = None
    best_noise = 1e-6
    
    # Try a range of noise levels to handle numerical stability vs exactness
    # For 200 points, 1e-6 to 1e-8 is usually safe.
    noise_candidates = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    
    for noise in noise_candidates:
        if time_check():
            break
            
        # Optimize hyperparameters for this noise level
        # Bounds for log_l: [-5, 5] -> l in [0.006, 148]
        # Bounds for log_sigma_f: [-5, 5] -> sigma_f in [0.006, 148]
        
        def neg_ll(params):
            return -log_likelihood(params, X_norm, y_norm, noise)
        
        # Use Nelder-Mead or L-BFGS-B
        # Nelder-Mead is robust for 2 parameters
        x0 = [0.0, 0.0] # l=1, sigma_f=1
        
        try:
            res = minimize(neg_ll, x0, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 200})
            if res.success or res.fun < -best_ll:
                # Check if the solution is valid
                val = -res.fun
                if val > best_ll:
                    best_ll = val
                    best_params = res.x
                    best_noise = noise
        except Exception:
            continue

    if best_params is not None:
        l_opt = np.exp(best_params[0])
        sigma_f_opt = np.exp(best_params[1])
        
        # Build the kernel matrix for training
        D2_train = cdist(X_norm, X_norm, metric='sqeuclidean')
        K_train = (sigma_f_opt ** 2) * np.exp(-0.5 * D2_train / (l_opt ** 2))
        K_train += best_noise * np.eye(n_samples)
        
        try:
            L_train = np.linalg.cholesky(K_train)
            alpha = np.linalg.solve(L_train, y_norm)
            
            # Create predict function
            def predict(X):
                X_pred = (X - X_mean) / X_std
                # Compute K(X, X_train)
                D2_test = cdist(X_pred, X_norm, metric='sqeuclidean')
                K_test = (sigma_f_opt ** 2) * np.exp(-0.5 * D2_test / (l_opt ** 2))
                
                # Predict mean: K_test @ alpha
                y_pred_norm = K_test @ alpha
                y_pred = y_pred_norm * y_scale + y_mean
                
                # Ensure finite
                y_pred[~np.isfinite(y_pred)] = 0.0
                return y_pred
            
            # Verify prediction quality on training data (sanity check)
            y_train_pred_norm = K_train @ alpha
            y_train_pred = y_train_pred_norm * y_scale + y_mean
            mse = np.mean((y_train_pred - y_train)**2)
            
            # If MSE is unreasonably high, GP might have failed to fit.
            # This can happen if hyperparameters are bad.
            if mse < 1e-2 * y_scale**2:
                return predict
        except Exception:
            pass

    # Fallback: If GP fails or is too slow, use a robust polynomial/feature fit
    # This is essentially the logic from the previous program, simplified.
    # We will use a high-degree polynomial fit with ridge regression to avoid overfitting
    # but allow flexibility.
    
    def gen_features(X):
        cols = [np.ones(len(X))]
        d = X.shape[1]
        # Linear
        for j in range(d):
            cols.append(X[:, j])
        # Quadratic
        for i in range(d):
            for j in range(i, d):
                cols.append(X[:, i] * X[:, j])
        # Cubic (only if d is small to avoid explosion)
        if d <= 3:
            for i in range(d):
                cols.append(X[:, i]**3)
                for j in range(i, d):
                    cols.append(X[:, i]**2 * X[:, j])
                    if i != j:
                        cols.append(X[:, i] * X[:, j]**2)
        return np.column_stack(cols)

    try:
        F_train = gen_features(X_train)
        F_train[~np.isfinite(F_train)] = 0.0
        
        # Normalize features
        F_means = F_train.mean(axis=0)
        F_stds = F_train.std(axis=0)
        F_stds[F_stds == 0] = 1.0
        F_norm = (F_train - F_means) / F_stds
        F_norm[:, 0] = 1.0
        
        # Ridge Regression
        lambda_ridge = 1e-2
        n_features_F = F_norm.shape[1]
        reg = lambda_ridge * np.eye(n_features_F)
        reg[0,0] = 0 # Don't regularize bias
        
        FtF = F_norm.T @ F_norm
        Fty = F_norm.T @ y_norm
        w = np.linalg.solve(FtF + reg, Fty)
        
        def predict_poly(X):
            F_pred = gen_features(X)
            F_pred[~np.isfinite(F_pred)] = 0.0
            F_pred_norm = (F_pred - F_means) / F_stds
            F_pred_norm[:, 0] = 1.0
            y_pred_norm = F_pred_norm @ w
            y_pred = y_pred_norm * y_scale + y_mean
            y_pred[~np.isfinite(y_pred)] = 0.0
            return y_pred
        
        # Check if this is better than a simple mean?
        # If y is constant, mean is best.
        return predict_poly

    except Exception:
        # Last resort: Return mean
        def predict_mean(X):
            return np.full(X.shape[0], y_mean)
        return predict_mean
