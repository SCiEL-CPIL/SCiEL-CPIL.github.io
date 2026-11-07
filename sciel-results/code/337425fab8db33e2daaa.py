import numpy as np
from scipy.linalg import cho_factor, cho_solve
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin
    
    X = np.asarray(X_train)
    y = np.asarray(y_train)
    
    # Ensure 1D handling
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    # Normalize X for better kernel conditioning
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std == 0] = 1.0
    X_norm = (X - x_mean) / x_std
    
    n = X_norm.shape[0]
    
    # Precompute squared distances for kernel evaluation
    # K_ij = exp(-0.5 * (x_i - x_j)^2 / l^2) for RBF, or Matern
    # We will use Matern 5/2
    
    def matern52_kernel(X1, X2, length_scale, signal_var):
        # X1, X2 are (N, D)
        # Compute squared distances
        # D_ij = ||x_i - x_j||^2
        x1_sq = np.sum(X1**2, axis=1)
        x2_sq = np.sum(X2**2, axis=1)
        dist_sq = x1_sq[:, None] + x2_sq[None, :] - 2 * X1 @ X2.T
        dist_sq = np.maximum(dist_sq, 0) # Numerical stability
        dist = np.sqrt(dist_sq)
        
        # Matern 5/2
        # k(r) = (1 + sqrt(5)r/l + 5r^2/(3l^2)) * exp(-sqrt(5)r/l)
        r = dist / length_scale
        kernel = (1 + np.sqrt(5)*r + (5/3)*r**2) * np.exp(-np.sqrt(5)*r)
        return signal_var * kernel

    best_predictor = None
    best_score = np.inf
    
    # Grid search over length scales
    # Heuristic: length scale should be related to data spacing
    # Avg spacing approx (max-min)/N
    x_range = X_norm.max(axis=0) - X_norm.min(axis=0)
    avg_spacing = np.mean(x_range) / n
    
    # Try a range of length scales relative to avg_spacing
    # Also try signal variance
    # Since y is noiseless, we want to fit it.
    # We'll use a fixed signal variance based on y's variance to normalize the kernel
    
    y_var = np.var(y)
    if y_var == 0:
        # Constant function
        return lambda X_test: np.full(X_test.shape[0], y[0])
        
    signal_var = y_var
    
    # Candidate length scales (in normalized units)
    # Normalized data has std ~ 1. So range is ~ 2-4.
    # Spacing is ~ 0.01-0.05.
    # Length scales: 0.1, 0.2, 0.5, 1.0, 2.0
    ls_candidates = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    
    # We want to find the GP that best predicts *unseen* data.
    # With noiseless data, all GPs interpolate.
    # The "best" one is subjective without a test set.
    # However, a common heuristic is to use a length scale that balances smoothness and fit.
    # Let's evaluate the "effective degrees of freedom" or just pick a robust one.
    # Actually, we can use Leave-One-Out (LOO) cross-validation to pick the best length scale.
    # LOO prediction error for GP is:
    # e_i = y_i - mu_i
    # mu_i = K_inv y
    # LOO error = e_i / (K_inv)_{ii}
    
    for ls in ls_candidates:
        if time.time() > deadline - 10:
            break
            
        K = matern52_kernel(X_norm, X_norm, ls, signal_var)
        # Add jitter for stability
        jitter = 1e-8 * np.trace(K) / n
        K += np.eye(n) * jitter
        
        try:
            L, low = cho_factor(K, lower=True, check_finite=False)
            alpha = cho_solve((L, low), y, check_finite=False)
            
            # Compute LOO errors
            # K_inv_y = alpha
            # Diagonal of K_inv is alpha / K_diag? No.
            # (K^{-1})_{ii} = alpha_i / K_{ii} is incorrect.
            # We need the diagonal of K^{-1}.
            # Let's solve K z = I to get K^{-1}.
            # This is O(N^3) again.
            # Alternatively, use the formula: 
            # mu_i = sum_j K_ij alpha_j
            # K_inv_ii = alpha_i / K_ii is NOT correct.
            # Actually, alpha = K^{-1} y.
            # The diagonal of K^{-1} can be found by solving K X = I.
            # But we can approximate or just use a simpler metric.
            
            # Let's just use the training log-likelihood as a proxy?
            # For noiseless data, likelihood is maximized when noise -> 0.
            # So it doesn't help distinguish length scales.
            
            # Let's stick to a fixed, reasonable length scale.
            # A length scale of 0.5 in normalized space (std=1) is a good general choice.
            # Or we can average predictions from multiple length scales?
            
        except Exception:
            continue
            
    # Strategy: Use an ensemble of GPs with different length scales?
    # Or just pick ls=0.5.
    
    # Let's try a different structural change: **Fourier Series Fit**.
    # For 1D periodic or smooth functions, Fourier series are excellent.
    # We can fit a Fourier series up to frequency M.
    # X is in [min, max]. Map to [0, 2*pi].
    # Basis: 1, cos(k*x), sin(k*x) for k=1..M.
    # This is linear regression, very fast.
    
    # Let's implement Fourier Series Fit as the primary method.
    # It handles oscillatory functions well.
    
    def fourier_features(X, M):
        # X is (N, 1)
        x = X.flatten()
        # Map to [0, 2*pi]
        x_min = x.min()
        x_max = x.max()
        if x_max == x_min:
            return np.ones((len(x), 1))
        x_scaled = (x - x_min) / (x_max - x_min) * 2 * np.pi
        
        cols = [np.ones(len(x))]
        for k in range(1, M+1):
            cols.append(np.cos(k * x_scaled))
            cols.append(np.sin(k * x_scaled))
        return np.column_stack(cols)
    
    # Also try Polynomial Fit up to degree 10
    def poly_features(X, deg):
        x = X.flatten()
        cols = [x**i for i in range(deg+1)]
        return np.column_stack(cols)
    
    # We will evaluate several models on the training data (perfect fit expected)
    # and select the one with the lowest "complexity" or just use an ensemble.
    # Since we don't have test data, we assume the simplest model that fits is best?
    # No, we need to generalize.
    
    # Let's use **Spline Interpolation** as the safest bet for noiseless 1D data.
    # It guarantees zero training error and smooth interpolation.
    
    from scipy.interpolate import CubicSpline
    
    try:
        # Sort by X
        idx = np.argsort(X.flatten())
        X_sorted = X[idx].flatten()
        y_sorted = y[idx]
        
        # Check for duplicate X values
        # If duplicates, average y
        unique_x, unique_indices, counts = np.unique(X_sorted, return_inverse=True, return_counts=True)
        if len(unique_x) < len(X_sorted):
             # Handle duplicates
             y_sum = np.zeros(len(unique_x))
             np.add.at(y_sum, unique_indices, y_sorted)
             y_sorted = y_sum / counts
             X_sorted = unique_x
             
        cs = CubicSpline(X_sorted, y_sorted)
        
        def predict(X_test):
            X_test = np.asarray(X_test)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            x_test = X_test.flatten()
            return cs(x_test)
            
        return predict
        
    except Exception:
        # Fallback to Linear Regression
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        w, *_ = np.linalg.lstsq(X_with_bias, y, rcond=None)
        def predict_linear(X_test):
            X_test = np.asarray(X_test)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            X_with_bias = np.column_stack([np.ones(X_test.shape[0]), X_test])
            return X_with_bias @ w
        return predict_linear
