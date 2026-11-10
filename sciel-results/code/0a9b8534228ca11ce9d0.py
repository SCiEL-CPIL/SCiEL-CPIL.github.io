# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Returns base positions for 64 tubes.
    """
    N = 8
    start_time = time.time()
    time_limit = 600.0
    margin = 60.0
    
    # Precompute slopes
    i_slopes = np.arange(N, dtype=np.float64) / N
    
    def union_length_1d(x, c):
        i = np.arange(N)
        starts = x + c * i / N
        ends = starts + 1.0 / N
        idx = np.argsort(starts)
        starts = starts[idx]
        ends = ends[idx]
        length = 0.0
        curr_start = starts[0]
        curr_end = ends[0]
        for k in range(1, N):
            if starts[k] <= curr_end:
                curr_end = max(curr_end, ends[k])
            else:
                length += curr_end - curr_start
                curr_start = starts[k]
                curr_end = ends[k]
        length += curr_end - curr_start
        return length

    def get_breakpoints_1d(coords):
        intercepts = np.empty(2 * N)
        intercepts[:N] = coords
        intercepts[N:] = coords + 1.0 / N
        ep_slopes = np.concatenate([i_slopes, i_slopes])
        
        i_idx, j_idx = np.triu_indices(2 * N, k=1)
        da = intercepts[j_idx] - intercepts[i_idx]
        db = ep_slopes[i_idx] - ep_slopes[j_idx]
        valid = np.abs(db) > 1e-12
        c_vals = da[valid] / db[valid]
        c_vals = c_vals[(c_vals >= 0.0) & (c_vals <= 1.0)]
        breakpoints = np.unique(np.sort(np.concatenate([[0.0], c_vals, [1.0]])))
        return breakpoints

    def exact_volume(params):
        x = params[:N]
        y = params[N:]
        
        bps_x = get_breakpoints_1d(x)
        bps_y = get_breakpoints_1d(y)
        
        bps = np.unique(np.sort(np.concatenate([bps_x, bps_y])))
        
        total_vol = 0.0
        for k in range(len(bps) - 1):
            c0 = bps[k]
            c1 = bps[k+1]
            if c1 - c0 < 1e-12:
                continue
            c_mid = (c0 + c1) / 2
            
            lx0 = union_length_1d(x, c0)
            lx1 = union_length_1d(x, c1)
            lxm = union_length_1d(x, c_mid)
            
            ly0 = union_length_1d(y, c0)
            ly1 = union_length_1d(y, c1)
            lym = union_length_1d(y, c_mid)
            
            integral = (c1 - c0) / 6.0 * (lx0*ly0 + 4*lxm*lym + lx1*ly1)
            total_vol += integral
            
        return total_vol

    # Initial guess: Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    x0 = keich.copy()
    y0 = keich.copy()
    params0 = np.concatenate([x0, y0])
    
    best_params = params0
    best_vol = exact_volume(params0)
    
    # Multi-restart
    rng_init = np.random.default_rng(42)
    initial_points = [params0]
    for _ in range(5):
        perturbed = keich + rng_init.normal(0, 0.05, size=N)
        initial_points.append(np.concatenate([perturbed, perturbed]))
    for _ in range(3):
        random_x = rng_init.uniform(-0.5, 1.5, size=N)
        random_y = rng_init.uniform(-0.5, 1.5, size=N)
        initial_points.append(np.concatenate([random_x, random_y]))
        
    for p0 in initial_points:
        if time.time() - start_time > time_limit - margin:
            break
        
        res = opt.minimize(exact_volume, p0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        if res.fun < best_vol:
            best_vol = res.fun
            best_params = res.x
            
    x_opt = best_params[:N]
    y_opt = best_params[N:]
    
    # Final polish
    if time.time() - start_time < time_limit - margin - 30:
        res_final = opt.minimize(exact_volume, best_params, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        if res_final.fun < best_vol:
            best_vol = res_final.fun
            best_params = res_final.x
            x_opt = best_params[:N]
            y_opt = best_params[N:]

    # Construct positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    if not np.all(np.isfinite(positions)):
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
        
    return positions

# EVOLVE-BLOCK-END
