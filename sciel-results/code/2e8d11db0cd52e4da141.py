import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    deadline = start_time + 590.0

    # Pre-compute quadrature points
    c_vals, w_vals = np.polynomial.legendre.leggauss(64)
    c_vals = 0.5 * (c_vals + 1.0)
    w_vals = 0.5 * w_vals

    def union_length(base, c, N):
        lefts = base + c * np.arange(N) / N
        rights = lefts + (np.arange(N) + 1) / N
        
        order = np.argsort(lefts)
        lefts_sorted = lefts[order]
        rights_sorted = rights[order]
        
        total_len = 0.0
        cur_l = lefts_sorted[0]
        cur_r = rights_sorted[0]
        for k in range(1, N):
            if lefts_sorted[k] <= cur_r:
                if rights_sorted[k] > cur_r:
                    cur_r = rights_sorted[k]
            else:
                total_len += cur_r - cur_l
                cur_l = lefts_sorted[k]
                cur_r = rights_sorted[k]
        total_len += cur_r - cur_l
        return total_len

    def objective(params):
        x_base = params[:N]
        y_base = params[N:]
        
        total_obj = 0.0
        for c, w in zip(c_vals, w_vals):
            lx = union_length(x_base, c, N)
            ly = union_length(y_base, c, N)
            total_obj += w * lx * ly
            
        return total_obj

    # Initial Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    x_base_init = keich.copy()
    y_base_init = keich.copy()
    initial_params = np.concatenate([x_base_init, y_base_init])
    
    best_params = initial_params.copy()
    best_obj = objective(best_params)

    # Optimization
    # 1. Nelder-Mead
    try:
        res_nm = minimize(
            objective,
            initial_params,
            method='Nelder-Mead',
            options={
                'maxiter': 20000,
                'xatol': 1e-9,
                'fatol': 1e-11,
                'adaptive': True,
                'initial_simplex': 0.1 * np.ones(2 * N)
            }
        )
        if res_nm.fun < best_obj:
            best_obj = res_nm.fun
            best_params = res_nm.x
    except Exception:
        pass

    # 2. Powell if time remains
    if time.time() < deadline - 60.0:
        try:
            res_powell = minimize(
                objective,
                best_params,
                method='Powell',
                options={
                    'maxiter': 20000,
                    'xatol': 1e-10,
                    'fatol': 1e-12
                }
            )
            if res_powell.fun < best_obj:
                best_obj = res_powell.fun
                best_params = res_powell.x
        except Exception:
            pass

    x_final = best_params[:N]
    y_final = best_params[N:]

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = x_final[i_flat]
    base_y = y_final[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    if not np.all(np.isfinite(positions)):
        return np.zeros((64, 2))
        
    return positions
