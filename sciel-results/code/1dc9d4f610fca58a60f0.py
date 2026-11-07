"""Rewritten program using Chebyshev spectral collocation for 1D symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 300.0  # Safety margin
    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    # Ensure X is 2D (n, d)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
        
    # We assume 1D input based on the task description, but handle general d if needed.
    # The task says "1 input(s)", so d=1.
    d = X_train.shape[1]
    
    if d != 1:
        # Fallback to polynomial if multi-dimensional (should not happen per contract)
        def features(X):
            X = np.asarray(X)
            if X.ndim == 1: X = X.reshape(-1, 1)
            cols = [np.ones(len(X))]
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        return lambda X: features(X) @ w

    # 1D Chebyshev Fit
    x_min = X_train[:, 0].min()
    x_max = X_train[:, 0].max()
    x_range = x_max - x_min
    if x_range == 0:
        x_range = 1.0
        x_min = 0.0
        
    def map_to_chebyshev(x):
        # Map from [x_min, x_max] to [-1, 1]
        return 2.0 * (x - x_min) / x_range - 1.0

    t_train = map_to_chebyshev(X_train[:, 0])
    
    # We will test a range of degrees to find the optimal one for noiseless data.
    # Since data is noiseless, we want the lowest degree that fits the data well.
    # However, if the function is complex, we need higher degrees.
    # Let's check degrees: 1, 5, 10, 20, 40, 80, 120, 160, 199
    candidate_degrees = [1, 5, 10, 20, 40, 80, 120, 160, 199]
    
    best_w = None
    best_degree = 1
    best_error = np.inf
    
    # Pre-compute Chebyshev values for training points for max degree
    # T_k(t) can be computed recursively or via cos(k * arccos(t))
    # Using cos is fine for 200 points.
    
    # Optimization: Compute the matrix for the highest degree first?
    # Or just loop. 200 points is small.
    
    for N in candidate_degrees:
        if time.time() > deadline - 5:
            break
            
        # Construct Chebyshev basis matrix C of shape (n, N+1)
        # C[i, k] = T_k(t_i)
        # T_0(t) = 1
        # T_1(t) = t
        # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
        
        n_points = len(t_train)
        C = np.zeros((n_points, N + 1))
        C[:, 0] = 1.0
        if N >= 1:
            C[:, 1] = t_train
        for k in range(2, N + 1):
            C[:, k] = 2.0 * t_train * C[:, k-1] - C[:, k-2]
            
        # Solve least squares
        # Since data is noiseless, we expect a very small residual.
        # Use lstsq for stability.
        try:
            w, residuals, rank, s = np.linalg.lstsq(C, y_train, rcond=None)
            y_pred = C @ w
            error = np.mean((y_pred - y_train) ** 2)
            
            # We want the smallest N that achieves a very low error.
            # If error is already extremely small (machine precision), we can stop.
            # Variance of y:
            var_y = np.var(y_train)
            if var_y == 0:
                rel_error = 0 if error == 0 else 1e10
            else:
                rel_error = error / var_y
                
            if best_w is None or error < best_error:
                best_error = error
                best_w = w
                best_degree = N
                
            # If relative error is negligible, we can assume we've captured the function
            # and higher degrees might just fit numerical noise or be redundant.
            # However, since it's noiseless, we can be strict.
            if rel_error < 1e-10:
                break
                
        except np.linalg.LinAlgError:
            continue

    # If no valid fit was found (unlikely), fallback to linear
    if best_w is None:
        # Simple linear regression
        A = np.column_stack([np.ones(len(X_train)), X_train[:, 0]])
        best_w, *_ = np.linalg.lstsq(A, y_train, rcond=None)
        
    # Capture variables for the lambda
    x_min_val = x_min
    x_range_val = x_range
    w_val = best_w
    N_val = best_degree
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        t = map_to_chebyshev(X[:, 0])
        
        # Construct Chebyshev basis for prediction points
        n_pred = len(t)
        C_pred = np.zeros((n_pred, N_val + 1))
        C_pred[:, 0] = 1.0
        if N_val >= 1:
            C_pred[:, 1] = t
        for k in range(2, N_val + 1):
            C_pred[:, k] = 2.0 * t * C_pred[:, k-1] - C_pred[:, k-2]
            
        return C_pred @ w_val

    return predict
# EVOLVE-BLOCK-END
