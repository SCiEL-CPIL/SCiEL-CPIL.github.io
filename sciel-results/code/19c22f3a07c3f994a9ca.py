import numpy as np
from scipy.optimize import minimize
import time

def _union_measure(intervals):
    if len(intervals) == 0:
        return 0.0
    sorted_intervals = intervals[np.argsort(intervals[:, 0])]
    total_measure = 0.0
    current_start = sorted_intervals[0, 0]
    current_end = sorted_intervals[0, 1]
    for i in range(1, len(sorted_intervals)):
        s = sorted_intervals[i, 0]
        e = sorted_intervals[i, 1]
        if s <= current_end:
            if e > current_end:
                current_end = e
        else:
            total_measure += current_end - current_start
            current_start = s
            current_end = e
    total_measure += current_end - current_start
    return total_measure

def _compute_volume_deterministic(x, y, num_c=500):
    N = len(x)
    c_vals = np.linspace(0, 1, num_c)
    vol = 0.0
    i_vals = np.arange(N) / N
    j_vals = np.arange(N) / N
    
    for c in c_vals:
        x_starts = x + c * i_vals
        x_ends = x_starts + 1.0 / N
        x_intervals = np.column_stack((x_starts, x_ends))
        mx = _union_measure(x_intervals)
        
        y_starts = y + c * j_vals
        y_ends = y_starts + 1.0 / N
        y_intervals = np.column_stack((y_starts, y_ends))
        my = _union_measure(y_intervals)
        
        vol += mx * my
    
    return vol * (1.0 / (num_c - 1))

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0 # 20s margin
    
    N = 8
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    x_curr = keich.copy()
    y_curr = keich.copy()
    
    # Alternating optimization
    for _ in range(10):
        if time.time() > deadline:
            break
            
        def obj_x(x):
            return _compute_volume_deterministic(x, y_curr, num_c=500)
            
        res_x = minimize(obj_x, x_curr, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11})
        if res_x.fun < obj_x(x_curr):
            x_curr = res_x.x
            
        if time.time() > deadline:
            break
            
        def obj_y(y):
            return _compute_volume_deterministic(x_curr, y, num_c=500)
            
        res_y = minimize(obj_y, y_curr, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11})
        if res_y.fun < obj_y(y_curr):
            y_curr = res_y.x

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_curr[i_flat]
    base_y = y_curr[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions
