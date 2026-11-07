import numpy as np

def fit(X_train, y_train):
    X_train = np.asarray(X_train).flatten()
    y_train = np.asarray(y_train).flatten()
    
    # Sort for potential stability, though not strictly needed for kernel
    sort_idx = np.argsort(X_train)
    X_sorted = X_train[sort_idx]
    y_sorted = y_train[sort_idx]
    
    # Handle duplicates
    unique_x, inverse_indices, counts = np.unique(X_sorted, return_inverse=True, return_counts=True)
    if len(unique_x) < len(X_sorted):
        y_sum = np.zeros_like(unique_x)
        np.add.at(y_sum, inverse_indices, y_sorted)
        y_avg = y_sum / counts
        X_fit = unique_x
        y_fit = y_avg
    else:
        X_fit = X_sorted
        y_fit = y_sorted
        
    N = len(X_fit)
    if N == 0:
        def predict(X):
            return np.zeros(len(np.asarray(X).flatten()))
        return predict
        
    # Determine a good bandwidth sigma
    # Use a grid of sigmas based on the range of X
    x_range = np.max(X_fit) - np.min(X_fit)
    if x_range == 0:
        # All X are same, return mean
        mean_y = np.mean(y_fit)
        def predict(X):
            return np.full(len(np.asarray(X).flatten()), mean_y)
        return predict
        
    # Candidate sigmas: range/100, range/20, range/10, range/5, range/2
    # Also consider median distance
    distances = np.diff(X_fit)
    median_dist = np.median(distances[distances > 0]) if len(distances[distances > 0]) > 0 else x_range / 10
    
    sigma_candidates = [x_range / 100, x_range / 20, x_range / 10, x_range / 5, median_dist]
    sigma_candidates = [s for s in sigma_candidates if s > 0]
    
    best_alpha = None
    best_mse = np.inf
    best_sigma = sigma_candidates[0]
    
    # Regularization parameter
    lam = 1e-6
    
    for sigma in sigma_candidates:
        # Compute kernel matrix
        # K[i, j] = exp(-(x_i - x_j)^2 / (2 * sigma^2))
        diff = X_fit[:, np.newaxis] - X_fit[np.newaxis, :]
        K = np.exp(-(diff ** 2) / (2 * sigma ** 2))
        
        # Add regularization
        K_reg = K + lam * np.eye(N)
        
        try:
            # Solve for alpha
            alpha = np.linalg.solve(K_reg, y_fit)
        except np.linalg.LinAlgError:
            continue
            
        # Calculate training MSE
        pred_train = K @ alpha
        mse = np.mean((pred_train - y_fit) ** 2)
        
        # We want low MSE, but also avoid overfitting. 
        # Since data is noiseless, low training MSE is good, but we must ensure generalization.
        # Kernel ridge with small lambda is nearly interpolation.
        # Let's pick the one with lowest MSE, but if they are all very low, pick larger sigma (smoother).
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_sigma = sigma
            
    # If all failed, fallback to linear
    if best_alpha is None:
        from scipy.interpolate import interp1d
        interp_func = interp1d(X_fit, y_fit, kind='linear', fill_value="extrapolate")
        def predict(X):
            X = np.asarray(X).flatten()
            return interp_func(X)
        return predict

    def predict(X):
        X = np.asarray(X).flatten()
        # Compute kernel between test points and training points
        diff = X[:, np.newaxis] - X_fit[np.newaxis, :]
        K_test = np.exp(-(diff ** 2) / (2 * best_sigma ** 2))
        return K_test @ best_alpha

    return predict
