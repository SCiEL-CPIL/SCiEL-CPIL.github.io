import numpy as np
import scipy.optimize
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    
    # Initial guess: Keich construction
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x = np.array(positions, dtype=np.float64)
    
    # Precompute constants for area calculation
    a = (np.arange(n) + 1) / n
    
    # Use a coarser grid for optimization
    M_coarse = 2000
    y_coarse = (np.arange(M_coarse) + 0.5) / M_coarse
    w_coarse = (1.0 - y_coarse) / n
    
    # Full grid for final evaluation
    M_full = 20000
    y_full = (np.arange(M_full) + 0.5) / M_full
    w_full = (1.0 - y_full) / n
    
    def calc_area_coarse(x_vals):
        L = x_vals[:, None] + a[:, None] * y_coarse[None, :]
        L_sorted = np.sort(L, axis=0)
        d = L_sorted[1:, :] - L_sorted[:-1, :]
        mins = np.minimum(d, w_coarse[None, :])
        union_lens = w_coarse + np.sum(mins, axis=0)
        return np.mean(union_lens)
    
    def calc_area_full(x_vals):
        L = x_vals[:, None] + a[:, None] * y_full[None, :]
        L_sorted = np.sort(L, axis=0)
        d = L_sorted[1:, :] - L_sorted[:-1, :]
        mins = np.minimum(d, w_full[None, :])
        union_lens = w_full + np.sum(mins, axis=0)
        return np.mean(union_lens)
    
    start_time = time.time()
    timeout = 590
    
    # Coordinate descent with coarse grid
    best_x = x.copy()
    best_area_coarse = calc_area_coarse(x)
    best_area_full = calc_area_full(x)
    
    step_sizes = np.ones(n) * 0.1
    min_step = 1e-6
    max_step = 1.0
    
    while time.time() - start_time < timeout:
        improved = False
        for i in range(n):
            if time.time() - start_time > timeout:
                break
            
            x_i_old = x[i]
            current_step = step_sizes[i]
            
            # Simple line search
            for _ in range(5):
                # Try positive and negative steps
                for sign in [1, -1]:
                    x_try = x.copy()
                    x_try[i] = x_i_old + sign * current_step
                    area_try = calc_area_coarse(x_try)
                    
                    if area_try < best_area_coarse:
                        x[i] = x_try[i]
                        best_area_coarse = area_try
                        improved = True
                        step_sizes[i] = max(current_step * 0.7, min_step)
                        break
                else:
                    current_step = max(current_step * 0.5, min_step)
                    continue
                break
            
            if x[i] != x_i_old:
                # Update best full area
                full_area = calc_area_full(x)
                if full_area < best_area_full:
                    best_area_full = full_area
                    best_x = x.copy()
    
    # Final check
    final_area = calc_area_full(best_x)
    if final_area < best_area_full:
        best_area_full = final_area
    
    return best_x
