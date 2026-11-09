# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    deadline = start_time + 600  # 600s limit, leave margin
    
    # Keich binary construction for initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    # Initial guess: Keich positions for x and y
    x0 = keich.copy()
    y0 = keich.copy()
    
    # Flatten to 16 variables, but fix x0=0, y0=0 to remove translation invariance
    # Variables: x1..x7, y1..y7 (14 variables)
    
    def unpack(params):
        x = np.zeros(N)
        x[1:] = params[:N-1]
        y = np.zeros(N)
        y[1:] = params[N-1:]
        return x, y

    def union_length(positions, c_vals, indices):
        M = len(c_vals)
        lengths = np.zeros(M)
        # Vectorized merge for N=8 is hard to write cleanly without loops in Python
        # but N is small.
        for k in range(M):
            c = c_vals[k]
            s = positions + c * indices / N
            e = s + 1.0 / N
            order = np.argsort(s)
            s_sorted = s[order]
            e_sorted = e[order]
            
            total_len = 0.0
            curr_start = s_sorted[0]
            curr_end = e_sorted[0]
            for i in range(1, N):
                if s_sorted[i] <= curr_end:
                    curr_end = max(curr_end, e_sorted[i])
                else:
                    total_len += curr_end - curr_start
                    curr_start = s_sorted[i]
                    curr_end = e_sorted[i]
            total_len += curr_end - curr_start
            lengths[k] = total_len
        return lengths

    def volume_fixed(params):
        x, y = unpack(params)
        c_vals = np.linspace(0, 1, 2000)
        indices = np.arange(N)
        
        lx = union_length(x, c_vals, indices)
        ly = union_length(y, c_vals, indices)
        
        v = np.trapz(lx * ly, c_vals)
        return v

    init_params = np.concatenate([x0[1:], y0[1:]])
    
    best_params = init_params
    best_vol = volume_fixed(init_params)
    
    # Optimization 1: Powell
    try:
        res = opt.minimize(volume_fixed, init_params, method='Powell', 
                           options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11})
        if res.fun < best_vol:
            best_vol = res.fun
            best_params = res.x
    except Exception as e:
        pass
    
    # Optimization 2: Nelder-Mead polish
    if time.time() < deadline - 60:
        try:
            res2 = opt.minimize(volume_fixed, best_params, method='Nelder-Mead',
                                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res2.fun < best_vol:
                best_vol = res2.fun
                best_params = res2.x
        except Exception as e:
            pass

    # Optimization 3: Coordinate Descent / Random Restart if time permits
    # Given the speed, we might just run Powell again from a slightly perturbed start?
    # Or just trust Powell.
    
    # Reconstruct positions
    x = np.zeros(N)
    x[1:] = best_params[:N-1]
    y = np.zeros(N)
    y[1:] = best_params[N-1:]
    
    # Construct the 64x2 array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x[i_flat]
    base_y = y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
