import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

# Precompute constants for speed
_N = 64
_M = 20000
_A_VALUES = ((np.arange(_N) + 1) / _N).reshape(-1, 1)
_Y_VALUES = ((np.arange(_M) + 0.5) / _M).reshape(1, -1)
_W_VALUES = (1.0 - _Y_VALUES) / _N

def _compute_area(x):
    """
    Computes the area of the union of the 64 triangles.
    x: array of shape (64,)
    """
    # L_i(y) = x_i + a_i * y
    L = x.reshape(-1, 1) + _A_VALUES * _Y_VALUES  # Shape (n, 20000)
    
    # Sort L along axis 0
    L_sorted = np.sort(L, axis=0)  # Shape (n, 20000)
    
    # Differences between consecutive sorted L's
    D = L_sorted[1:, :] - L_sorted[:-1, :]  # Shape (n-1, 20000)
    
    # Union length at each y: w + sum(min(D_j, w))
    min_diffs = np.minimum(D, _W_VALUES)
    
    # Sum over triangles (axis 0)
    union_lengths = _W_VALUES[0, :] + np.sum(min_diffs, axis=0)  # Shape (20000,)
    
    # Area is mean of union lengths
    area = np.mean(union_lengths)
    return area

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    
    # 1. Initialize with Keich Construction
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(positions, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    # 2. L-BFGS-B Optimization
    # The objective is convex, so L-BFGS-B should converge well.
    res = minimize(_compute_area, x0, method='L-BFGS-B', 
                   options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
    x_best = res.x
    area_best = res.fun
    
    if time.time() > deadline:
        return x_best
    
    # 3. Coordinate Descent Polishing
    # The function is piecewise-linear convex. Coordinate descent is effective.
    # We iterate through coordinates, optimizing each one using minimize_scalar.
    
    improved = True
    max_rounds = 100
    round_num = 0
    while improved and (time.time() < deadline) and round_num < max_rounds:
        improved = False
        round_num += 1
        
        # Shrink window as we converge
        if round_num < 5:
            window = 0.5
        elif round_num < 15:
            window = 0.1
        else:
            window = 0.01
            
        for i in range(n):
            if time.time() > deadline:
                return x_best
                
            # Define 1D function for coordinate i
            def f_1d(xi, _i=i, _x=x_best):
                x_temp = _x.copy()
                x_temp[_i] = xi
                return _compute_area(x_temp)
            
            # Search in a window around current x_best[i]
            res_i = minimize_scalar(f_1d, bounds=(x_best[i] - window, x_best[i] + window), 
                                    method='bounded', options={'xatol': 1e-10})
            
            if res_i.success:
                if abs(res_i.x - x_best[i]) > 1e-12:
                    x_best[i] = res_i.x
                    improved = True
                if res_i.fun < area_best:
                    area_best = res_i.fun
    
    # 4. Final fine-tuning with very small steps if time permits
    if time.time() < deadline - 20:
        for _ in range(2):
            if time.time() > deadline:
                break
            for i in range(n):
                if time.time() > deadline:
                    break
                def f_1d_fine(xi, _i=i, _x=x_best):
                    x_temp = _x.copy()
                    x_temp[_i] = xi
                    return _compute_area(x_temp)
                
                res_i = minimize_scalar(f_1d_fine, bounds=(x_best[i] - 0.001, x_best[i] + 0.001), 
                                        method='bounded', options={'xatol': 1e-12})
                if res_i.success and res_i.fun < area_best:
                    x_best[i] = res_i.x
                    area_best = res_i.fun

    return x_best
