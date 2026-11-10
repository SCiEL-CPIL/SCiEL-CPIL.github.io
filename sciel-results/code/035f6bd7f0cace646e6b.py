import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Minimizes the area of the union of 64 triangles using scipy.optimize.
    """
    n = 64
    k = 20000
    # Precompute heights
    y = (np.arange(k) + 0.5) / k
    w = (1.0 - y) / n
    
    # Precompute a_i
    a = (np.arange(1, n + 1)) / n
    
    def area(x):
        """
        Computes the area of the union of triangles.
        x: array of shape (n,)
        """
        # L_i(y) = x_i + a_i * y
        # We need L for all i and all y.
        # x is (n,), a is (n,), y is (k,)
        # L[i, m] = x[i] + a[i] * y[m]
        # Using broadcasting: x[:, None] + a[:, None] * y[None, :]
        L = x[:, None] + a[:, None] * y[None, :]
        
        # Sort L along axis 1 (for each y, sort the L_i values)
        L_sorted = np.sort(L, axis=1)
        
        # Calculate gaps between consecutive sorted L values
        # d[i, m] = L_sorted[i+1, m] - L_sorted[i, m]
        d = np.diff(L_sorted, axis=1)
        
        # Union length at height m is w[m] + sum(min(d_i, w[m]))
        # d is (k, n-1), w is (k,)
        # min(d, w) needs broadcasting: d vs w[None, :]
        min_d_w = np.minimum(d, w[None, :])
        
        # Sum over i for each m
        union_len = w + np.sum(min_d_w, axis=1)
        
        # Area is mean of union lengths
        return np.mean(union_len)

    # Initialization: Aligned positions
    # x_i = 0.5 - a_i
    x0 = 0.5 - a
    
    # Initial area
    best_x = x0.copy()
    best_area = area(best_x)
    
    # Optimization
    # Use Nelder-Mead. It is robust for non-smooth functions.
    # We set a deadline guard.
    start_time = time.time()
    deadline = start_time + 600.0 - 30.0 # 30s safety margin
    
    # We can try multiple methods or restarts if time permits.
    # Nelder-Mead is a good start.
    try:
        res = minimize(
            area, 
            best_x, 
            method='Nelder-Mead', 
            options={
                'xatol': 1e-8, 
                'fatol': 1e-10, 
                'maxiter': 20000, 
                'maxfev': 20000,
                'adaptive': True
            }
        )
        
        if res.fun < best_area:
            best_area = res.fun
            best_x = res.x
            
        # Check time
        if time.time() < deadline:
            # Try Powell method as a second pass or refinement
            # Powell can sometimes find better minima in high dimensions
            res2 = minimize(
                area, 
                best_x, 
                method='Powell', 
                options={
                    'xtol': 1e-8, 
                    'ftol': 1e-10, 
                    'maxiter': 5000, 
                    'maxfev': 5000
                }
            )
            
            if res2.fun < best_area:
                best_area = res2.fun
                best_x = res2.x
                
    except Exception as e:
        # If optimization fails, return the initial guess
        pass
        
    # Final check for validity
    if not np.all(np.isfinite(best_x)):
        best_x = x0
        
    return best_x
