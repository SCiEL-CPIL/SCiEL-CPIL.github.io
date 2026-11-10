import numpy as np
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    a = (np.arange(n) + 1) / n
    
    # Pre-compute y values for different resolutions
    M_fast = 2000
    M_full = 20000
    y_fast = (np.arange(M_fast) + 0.5) / M_fast
    y_full = (np.arange(M_full) + 0.5) / M_full
    
    def _calc_area(x, y_vals, M):
        # x is shape (n,)
        # L_i(y) = x_i + a_i * y
        # L is shape (M, n)
        L = x[None, :] + a[None, :] * y_vals[:, None]
        
        # Sort L along axis 1
        L_sorted = np.sort(L, axis=1)
        
        # Differences between consecutive sorted elements
        diffs = np.diff(L_sorted, axis=1)
        
        # w(y) shape (M,)
        w = (1 - y_vals) / n
        
        # min(diffs, w)
        min_val = np.minimum(diffs, w[:, None])
        
        # Union length = w + sum(min_val, axis=1)
        union_lengths = w + np.sum(min_val, axis=1)
        
        # Area = mean(union_lengths)
        return np.mean(union_lengths)

    def area_fast(x):
        return _calc_area(x, y_fast, M_fast)
        
    def area_full(x):
        return _calc_area(x, y_full, M_full)

    best_x = None
    best_area_full = np.inf
    
    rng = np.random.default_rng(42)
    
    # Strategy: 
    # 1. Use Powell on area_fast to find a good basin.
    # 2. Refine with Powell on area_full (small budget) to respect the exact metric.
    
    for i in range(10):
        # Initial guess: 
        # Heuristic: x_i = -a_i creates a "fan" where they meet at (0,1).
        # Add some noise to break symmetry.
        if i == 0:
            x0 = -a.copy()
        else:
            x0 = -a.copy() + rng.normal(0, 0.1, n)
            
        # Phase 1: Coarse optimization
        res1 = minimize(
            area_fast, 
            x0, 
            method='Powell', 
            options={
                'maxiter': 500, 
                'maxfev': 2000,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'disp': False
            }
        )
        
        x_candidate = res1.x
        
        # Phase 2: Fine optimization
        # Use a smaller budget here as it's slower
        res2 = minimize(
            area_full, 
            x_candidate, 
            method='Powell', 
            options={
                'maxiter': 100, 
                'maxfev': 200,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'disp': False
            }
        )
        
        x_final = res2.x
        area_val = area_full(x_final)
        
        if area_val < best_area_full:
            best_area_full = area_val
            best_x = x_final
            
    return best_x.astype(np.float64)
