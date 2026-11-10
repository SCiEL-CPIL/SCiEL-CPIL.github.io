import numpy as np
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    a = (np.arange(n) + 1) / n
    M = 20000
    y_vals = (np.arange(M) + 0.5) / M
    
    def area(x):
        # x is shape (n,)
        # L_i(y) = x_i + a_i * y
        # We need to compute the union length for each y
        # L is shape (M, n)
        L = x[None, :] + a[None, :] * y_vals[:, None]
        
        # Sort L along axis 1
        # np.sort is stable and fast
        L_sorted = np.sort(L, axis=1)
        
        # Differences between consecutive sorted elements
        # diff shape (M, n-1)
        diffs = np.diff(L_sorted, axis=1)
        
        # w(y) shape (M,)
        w = (1 - y_vals) / n
        
        # min(diffs, w)
        # We need to sum over axis 1
        # min_val shape (M, n-1)
        min_val = np.minimum(diffs, w[:, None])
        
        # Union length = w + sum(min_val, axis=1)
        union_lengths = w + np.sum(min_val, axis=1)
        
        # Area = mean(union_lengths)
        return np.mean(union_lengths)

    # Initial guess
    # A simple heuristic: spread them out? 
    # Or use the Keich construction as a starting point? 
    # The directive says "do not reuse the current designs". 
    # Using Keich as a *starting point* for an optimizer is arguably reusing the design's output.
    # Let's use a random start or a simple linear spread.
    # Random start might get stuck in local minima.
    # Let's try a few random restarts or a deterministic pseudo-random start.
    
    # To ensure we get a good result, let's use a small number of restarts.
    # Nelder-Mead is stochastic in its path but deterministic given a start.
    
    best_x = None
    best_area = np.inf
    
    # Try a few random initializations
    rng = np.random.default_rng(42)
    
    for _ in range(5):
        x0 = rng.uniform(-0.5, 0.5, n)
        
        # Nelder-Mead options
        # maxiter and maxfev to prevent timeout
        result = minimize(
            area, 
            x0, 
            method='Nelder-Mead', 
            options={
                'maxiter': 500, 
                'maxfev': 500,
                'xatol': 1e-5,
                'fatol': 1e-8
            }
        )
        
        if result.fun < best_area:
            best_area = result.fun
            best_x = result.x

    return best_x.astype(np.float64)
