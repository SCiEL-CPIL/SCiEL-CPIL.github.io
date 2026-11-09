import numpy as np
from scipy.optimize import minimize
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    W = 1.0 / N
    start_time = time.time()
    TIME_LIMIT = 600.0

    # Precompute indices
    i_idx, j_idx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()

    def union_area_fast(x_starts, y_starts):
        """
        Compute the area of the union of 64 axis-aligned squares of side W.
        x_starts, y_starts: shape (64,)
        Uses sweep line with vectorized interval merging.
        """
        # Create events: (x, type, y0, y1)
        x0 = x_starts
        x1 = x_starts + W
        y0 = y_starts
        y1 = y_starts + W
        
        # Build event list as arrays for sorting
        # We need to sort by x, then process
        # For 64 squares, 128 events - still small enough for Python sort
        
        events = np.empty((128, 4), dtype=np.float64)
        events[0::2, 0] = x0
        events[0::2, 1] = 1.0
        events[0::2, 2] = y0
        events[0::2, 3] = y1
        events[1::2, 0] = x1
        events[1::2, 1] = -1.0
        events[1::2, 2] = y0
        events[1::2, 3] = y1
        
        # Sort by x coordinate
        sort_idx = np.argsort(events[:, 0], kind='stable')
        events = events[sort_idx]
        
        # Sweep line
        active_y0 = []
        active_y1 = []
        total_area = 0.0
        prev_x = events[0, 0]
        
        i = 0
        n_events = 128
        while i < n_events:
            x_curr = events[i, 0]
            dx = x_curr - prev_x
            
            if dx > 1e-12 and len(active_y0) > 0:
                # Merge intervals
                arr = np.array(list(zip(active_y0, active_y1)))
                arr = arr[np.argsort(arr[:, 0])]
                merged_start = arr[0, 0]
                merged_end = arr[0, 1]
                union_len = 0.0
                for m in range(1, len(arr)):
                    if arr[m, 0] <= merged_end:
                        if arr[m, 1] > merged_end:
                            merged_end = arr[m, 1]
                    else:
                        union_len += merged_end - merged_start
                        merged_start = arr[m, 0]
                        merged_end = arr[m, 1]
                union_len += merged_end - merged_start
                total_area += union_len * dx
            
            # Process all events at this x
            while i < n_events and abs(events[i, 0] - x_curr) < 1e-12:
                typ = events[i, 1]
                y0_val = events[i, 2]
                y1_val = events[i, 3]
                if typ == 1.0:
                    active_y0.append(y0_val)
                    active_y1.append(y1_val)
                else:
                    # Find and remove matching interval
                    for j in range(len(active_y0)):
                        if abs(active_y0[j] - y0_val) < 1e-12 and abs(active_y1[j] - y1_val) < 1e-12:
                            active_y0.pop(j)
                            active_y1.pop(j)
                            break
                i += 1
            
            prev_x = x_curr
        
        return total_area

    def calc_volume(x_base, y_base):
        """
        Estimate volume by integrating area over c using trapezoidal rule.
        x_base, y_base: shape (8,)
        """
        c_vals = np.linspace(0, 1, 32)
        areas = np.zeros(len(c_vals))
        
        for idx, c in enumerate(c_vals):
            x_starts = x_base[i_flat] + c * i_flat / N
            y_starts = y_base[j_flat] + c * j_flat / N
            areas[idx] = union_area_fast(x_starts, y_starts)
            
        return np.trapezoid(areas, c_vals)

    # Step 1: Find a good separable base Z using multiple restarts
    def obj_separable(params):
        return calc_volume(params, params)
    
    best_Z = None
    best_vol = np.inf
    
    # Try several initial guesses
    initial_guesses = [
        np.zeros(N),
        np.linspace(-0.5, 0.5, N),
        np.random.uniform(-0.3, 0.3, N),
        np.random.uniform(-0.3, 0.3, N),
    ]
    
    for x0 in initial_guesses:
        if time.time() - start_time > TIME_LIMIT - 150: break
        
        res = minimize(obj_separable, x0, method='Nelder-Mead', 
                      options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 3000})
        if res.fun < best_vol:
            best_vol = res.fun
            best_Z = res.x
            
    if best_Z is None:
        best_Z = np.zeros(N)
        
    # Step 2: Coordinate Descent for X and Y with L-BFGS-B
    best_X = best_Z.copy()
    best_Y = best_Z.copy()
    
    def obj_X(params):
        return calc_volume(params, best_Y)
    
    def obj_Y(params):
        return calc_volume(best_X, params)
    
    for _ in range(8):
        if time.time() - start_time > TIME_LIMIT - 40: break
        
        # Optimize X
        res_x = minimize(obj_X, best_X, method='Nelder-Mead', 
                        options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 2000})
        if res_x.fun < calc_volume(best_X, best_Y):
            best_X = res_x.x
            
        # Optimize Y
        res_y = minimize(obj_Y, best_Y, method='Nelder-Mead', 
                        options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 2000})
        if res_y.fun < calc_volume(best_X, best_Y):
            best_Y = res_y.x

    # Step 3: Final polish with tighter tolerances
    if time.time() - start_time < TIME_LIMIT - 20:
        res_x_final = minimize(obj_X, best_X, method='Nelder-Mead', 
                              options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 5000})
        if res_x_final.fun < calc_volume(best_X, best_Y):
            best_X = res_x_final.x
        
        res_y_final = minimize(obj_Y, best_Y, method='Nelder-Mead', 
                              options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 5000})
        if res_y_final.fun < calc_volume(best_X, best_Y):
            best_Y = res_y_final.x

    # Construct 64 positions
    base_x = best_X[i_flat]
    base_y = best_Y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
