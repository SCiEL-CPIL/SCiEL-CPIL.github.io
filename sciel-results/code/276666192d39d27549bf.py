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
    M_full = 20000
    M_inner = 2000  # Coarser grid for inner optimization
    
    y_full = (np.arange(M_full) + 0.5) / M_full
    y_inner = (np.arange(M_inner) + 0.5) / M_inner
    
    # Precompute A = a[:, None] * y[None, :] for both grids
    A_full = a[:, None] * y_full[None, :]
    A_inner = a[:, None] * y_inner[None, :]
    w_full = (1.0 - y_full) / n
    w_inner = (1.0 - y_inner) / n
    
    def calc_area(x_vals, A=None, w=None, M=None):
        if A is None:
            A = A_full
            w = w_full
            M = M_full
        # L_i,m = x_i + A_i,m
        L = x_vals[:, None] + A
        L_sorted = np.sort(L, axis=0)
        d = L_sorted[1:, :] - L_sorted[:-1, :]
        mins = np.minimum(d, w[None, :])
        union_lens = w + np.sum(mins, axis=0)
        area = np.mean(union_lens)
        return area
    
    def calc_area_inner(x_vals):
        return calc_area(x_vals, A_inner, w_inner, M_inner)
    
    def calc_area_full(x_vals):
        return calc_area(x_vals, A_full, w_full, M_full)
    
    start_time = time.time()
    timeout = 590  # Safety margin before 630s kill
    
    # Coordinate Descent with inner grid for speed
    sweep = 0
    while time.time() - start_time < timeout:
        for i in range(n):
            if time.time() - start_time > timeout - 30:
                break
            
            x_i_old = x[i]
            
            # Objective function for 1D optimization using inner grid
            def obj_1d(xi):
                x_new = x.copy()
                x_new[i] = xi
                return calc_area_inner(x_new)
            
            # Optimize xi
            res = scipy.optimize.minimize_scalar(
                obj_1d, 
                bounds=(x_i_old - 0.5, x_i_old + 0.5), 
                method='bounded', 
                options={'xatol': 1e-5}
            )
            
            if res.success:
                x[i] = res.x
        
        sweep += 1
    
    # Final refinement with full grid
    if time.time() - start_time < timeout - 60:
        for i in range(n):
            if time.time() - start_time > timeout - 60:
                break
            x_i_old = x[i]
            
            def obj_1d_full(xi):
                x_new = x.copy()
                x_new[i] = xi
                return calc_area_full(x_new)
            
            res = scipy.optimize.minimize_scalar(
                obj_1d_full, 
                bounds=(x_i_old - 0.3, x_i_old + 0.3), 
                method='bounded', 
                options={'xatol': 1e-7}
            )
            
            if res.success:
                x[i] = res.x
    
    return x
