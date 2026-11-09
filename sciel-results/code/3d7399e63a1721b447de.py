import numpy as np
from scipy.optimize import minimize
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    W = 1.0 / N
    start_time = time.time()
    TIME_LIMIT = 600.0

    # Helper to get trapezoid function
    try:
        _trapz = np.trapezoid
    except AttributeError:
        _trapz = np.trapz

    def union_area(x_starts, y_starts):
        """
        Compute the area of the union of 64 axis-aligned squares of side W.
        x_starts, y_starts: shape (64,)
        """
        events = []
        for k in range(64):
            x0 = x_starts[k]
            x1 = x0 + W
            y0 = y_starts[k]
            y1 = y0 + W
            events.append((x0, 1, y0, y1))
            events.append((x1, -1, y0, y1))
        
        events.sort(key=lambda e: e[0])
        
        active_intervals = []
        total_area = 0.0
        prev_x = events[0][0]
        
        i = 0
        while i < len(events):
            x_curr = events[i][0]
            dx = x_curr - prev_x
            if dx > 1e-12:
                if active_intervals:
                    active_intervals.sort(key=lambda t: t[0])
                    union_len = 0.0
                    cur_start, cur_end = active_intervals[0]
                    for y_s, y_e in active_intervals[1:]:
                        if y_s <= cur_end:
                            if y_e > cur_end:
                                cur_end = y_e
                        else:
                            union_len += cur_end - cur_start
                            cur_start, cur_end = y_s, y_e
                    union_len += cur_end - cur_start
                    total_area += union_len * dx
            
            while i < len(events) and events[i][0] == x_curr:
                _, typ, y0, y1 = events[i]
                if typ == 1:
                    active_intervals.append((y0, y1))
                else:
                    try:
                        active_intervals.remove((y0, y1))
                    except ValueError:
                        pass
                i += 1
            
            prev_x = x_curr
            
        return total_area

    def calc_volume(x_base, y_base):
        """
        Estimate volume by integrating area over c.
        x_base, y_base: shape (8,)
        """
        i_idx, j_idx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        i_flat = i_idx.flatten()
        j_flat = j_idx.flatten()
        
        c_vals = np.linspace(0, 1, 20)
        areas = np.zeros(len(c_vals))
        
        for idx, c in enumerate(c_vals):
            x_starts = x_base[i_flat] + c * i_flat / N
            y_starts = y_base[j_flat] + c * j_flat / N
            areas[idx] = union_area(x_starts, y_starts)
            
        return _trapz(areas, c_vals)

    # Step 1: Find a good separable base Z
    def obj_separable(params):
        return calc_volume(params, params)
    
    best_Z = None
    best_vol = np.inf
    
    for _ in range(3):
        if time.time() - start_time > TIME_LIMIT - 120: break
        
        x0 = np.zeros(N) if _ == 0 else np.random.uniform(-0.5, 0.5, N)
        res = minimize(obj_separable, x0, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 2000})
        if res.fun < best_vol:
            best_vol = res.fun
            best_Z = res.x
            
    if best_Z is None:
        best_Z = np.zeros(N)
        
    # Step 2: Coordinate Descent for X and Y
    best_X = best_Z.copy()
    best_Y = best_Z.copy()
    
    for _ in range(5):
        if time.time() - start_time > TIME_LIMIT - 30: break
        
        # Optimize X
        def obj_X(params):
            return calc_volume(params, best_Y)
        res_x = minimize(obj_X, best_X, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 1000})
        if res_x.fun < calc_volume(best_X, best_Y):
            best_X = res_x.x
            
        # Optimize Y
        def obj_Y(params):
            return calc_volume(best_X, params)
        res_y = minimize(obj_Y, best_Y, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 1000})
        if res_y.fun < calc_volume(best_X, best_Y):
            best_Y = res_y.x

    # Construct 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_X[i_flat]
    base_y = best_Y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
