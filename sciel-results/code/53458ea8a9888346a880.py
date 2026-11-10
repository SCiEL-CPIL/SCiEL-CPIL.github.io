# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    deadline = start_time + 590.0

    # Fast union length calculation for N=8
    def union_length_fast(starts, ends):
        # starts, ends: np.array shape (8,)
        idx = np.argsort(starts)
        s = starts[idx]
        e = ends[idx]
        
        # Merge
        # Unrolled for speed? No, loop is fine for N=8
        merged_s = np.empty(N)
        merged_e = np.empty(N)
        cnt = 0
        cs = s[0]
        ce = e[0]
        for i in range(1, N):
            if s[i] <= ce:
                if e[i] > ce:
                    ce = e[i]
            else:
                merged_s[cnt] = cs
                merged_e[cnt] = ce
                cnt += 1
                cs = s[i]
                ce = e[i]
        merged_s[cnt] = cs
        merged_e[cnt] = ce
        cnt += 1
        return np.sum(merged_e[:cnt] - merged_s[:cnt])

    def compute_volume(u, v, num_c=200):
        c_vals = np.linspace(0, 1, num_c)
        i_factors = np.arange(N) / N
        j_factors = np.arange(N) / N
        width = 1.0 / N
        
        total_vol = 0.0
        for c in c_vals:
            x_starts = u + c * i_factors
            x_ends = x_starts + width
            x_len = union_length_fast(x_starts, x_ends)
            
            y_starts = v + c * j_factors
            y_ends = y_starts + width
            y_len = union_length_fast(y_starts, y_ends)
            
            total_vol += x_len * y_len
            
        return total_vol / num_c

    # Initial guess: Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    u_init = keich.copy()
    v_init = keich.copy()
    x0 = np.concatenate([u_init, v_init])
    
    best_params = x0.copy()
    best_vol = compute_volume(u_init, v_init, 500)
    
    # Bounds for differential evolution
    # The positions can vary, but the relative shifts matter.
    # We can restrict the range to [-1, 2] to keep the box small.
    bounds = [(-1.0, 2.0)] * 16
    
    # Global Optimization
    if time.time() < deadline - 120:
        def cost_func(params):
            u = params[:N]
            v = params[N:]
            return compute_volume(u, v, 100) # Lower precision for DE
        
        try:
            res = differential_evolution(
                cost_func, 
                bounds, 
                seed=42, 
                maxiter=100, 
                tol=1e-4, 
                polish=False, 
                init='sobol'
            )
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x
        except Exception:
            pass

    # Local Polish
    if time.time() < deadline - 60:
        def cost_func_high(params):
            u = params[:N]
            v = params[N:]
            return compute_volume(u, v, 500)
            
        try:
            res = minimize(
                cost_func_high, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x
        except Exception:
            pass

    # Final Polish with even higher precision
    if time.time() < deadline - 30:
        def cost_func_final(params):
            u = params[:N]
            v = params[N:]
            return compute_volume(u, v, 2000)
            
        try:
            res = minimize(
                cost_func_final, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x
        except Exception:
            pass

    best_u = best_params[:N]
    best_v = best_params[N:]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_u[i_flat]
    base_y = best_v[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
