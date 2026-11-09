import numpy as np
import time
from scipy.optimize import minimize

def _merge_interval_lengths(intervals):
    """Merge overlapping 1D intervals and return total covered length."""
    if len(intervals) == 0:
        return 0.0
    intervals = intervals[np.argsort(intervals[:, 0])]
    total = 0.0
    cur_l, cur_r = intervals[0, 0], intervals[0, 1]
    for k in range(1, len(intervals)):
        l, r = intervals[k, 0], intervals[k, 1]
        if l <= cur_r:
            if r > cur_r:
                cur_r = r
        else:
            total += cur_r - cur_l
            cur_l, cur_r = l, r
    total += cur_r - cur_l
    return total

def compute_union_area_at_c(c, x_positions, y_positions, N=8):
    """
    Compute the area of the union of rectangles at height c using interval merging.
    x_positions: array of shape (N,) for x-base positions (one per i)
    y_positions: array of shape (N,) for y-base positions (one per j)
    """
    i = np.arange(N)
    x_left = x_positions + c * i / N
    x_right = x_left + 1.0 / N
    j = np.arange(N)
    y_left = y_positions + c * j / N
    y_right = y_left + 1.0 / N
    
    x_intervals = np.column_stack([x_left, x_right])
    y_intervals = np.column_stack([y_left, y_right])
    
    x_len = _merge_interval_lengths(x_intervals)
    y_len = _merge_interval_lengths(y_intervals)
    
    return x_len * y_len

def compute_volume(x_positions, y_positions, N=8, num_c=64):
    """
    Estimate the volume by integrating the area over c in [0,1] using Gauss-Legendre quadrature.
    """
    c_vals, w_vals = np.polynomial.legendre.leggauss(num_c)
    # Map from [-1,1] to [0,1]
    c_vals = 0.5 * (c_vals + 1.0)
    w_vals = 0.5 * w_vals
    areas = np.array([compute_union_area_at_c(c, x_positions, y_positions, N) for c in c_vals])
    return np.sum(w_vals * areas)

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    max_time = 590.0  # Safety margin

    # Initial Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    x_positions = keich.copy()
    y_positions = keich.copy()
    
    best_vol = compute_volume(x_positions, y_positions)
    best_x = x_positions.copy()
    best_y = y_positions.copy()
    
    rng = np.random.default_rng(42)
    
    def objective(params):
        """Objective function for scipy optimizer: returns volume."""
        x = params[:N]
        y = params[N:]
        return compute_volume(x, y)
    
    def local_search(x_init, y_init, max_iters=2000):
        x = x_init.copy()
        y = y_init.copy()
        curr_vol = compute_volume(x, y)
        best_x_local = x.copy()
        best_y_local = y.copy()
        best_vol_local = curr_vol
        step = 0.05
        
        for it in range(max_iters):
            if time.time() - start_time > max_time - 10:
                break
            
            improved = False
            # Try moving each x coordinate
            for i in range(N):
                for sign in [-1, 1]:
                    x_new = x.copy()
                    x_new[i] += sign * step
                    v = compute_volume(x_new, y)
                    if v < curr_vol:
                        x = x_new
                        curr_vol = v
                        improved = True
                        if curr_vol < best_vol_local:
                            best_vol_local = curr_vol
                            best_x_local = x.copy()
                            best_y_local = y.copy()
                        break
            
            # Try moving each y coordinate
            for j in range(N):
                for sign in [-1, 1]:
                    y_new = y.copy()
                    y_new[j] += sign * step
                    v = compute_volume(x, y_new)
                    if v < curr_vol:
                        y = y_new
                        curr_vol = v
                        improved = True
                        if curr_vol < best_vol_local:
                            best_vol_local = curr_vol
                            best_x_local = x.copy()
                            best_y_local = y.copy()
                        break
            
            if not improved:
                # Try smaller step
                step *= 0.7
                if step < 1e-6:
                    break
            else:
                # Keep step or increase slightly
                step = min(step * 1.1, 0.1)
        
        return best_x_local, best_y_local, best_vol_local
    
    # Run local search from seed
    x_cand, y_cand, vol_cand = local_search(x_positions, y_positions)
    if vol_cand < best_vol:
        best_x = x_cand
        best_y = y_cand
        best_vol = vol_cand
    
    # Use scipy.optimize.minimize for gradient-free optimization (Nelder-Mead)
    def scipy_optimize(x_init, y_init):
        """Run Nelder-Mead optimization on the 2N-dimensional parameter vector."""
        x0 = np.concatenate([x_init, y_init])
        
        # Check time before starting
        if time.time() - start_time > max_time - 30:
            return x_init, y_init, compute_volume(x_init, y_init)
        
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            x_opt = result.x[:N]
            y_opt = result.x[N:]
            vol_opt = result.fun
            return x_opt, y_opt, vol_opt
        except Exception:
            return x_init, y_init, compute_volume(x_init, y_init)
    
    # Run scipy optimization from seed
    if time.time() - start_time < max_time - 30:
        x_cand, y_cand, vol_cand = scipy_optimize(best_x, best_y)
        if vol_cand < best_vol:
            best_x = x_cand
            best_y = y_cand
            best_vol = vol_cand
    
    # Random restarts with both local search and scipy optimization
    restarts = 0
    while time.time() - start_time < max_time - 30:
        x_init = rng.uniform(-0.5, 0.5, N)
        y_init = rng.uniform(-0.5, 0.5, N)
        
        # Alternate between local search and scipy optimization
        if restarts % 2 == 0:
            x_cand, y_cand, vol_cand = local_search(x_init, y_init, max_iters=300)
        else:
            x_cand, y_cand, vol_cand = scipy_optimize(x_init, y_init)
        
        if vol_cand < best_vol:
            best_x = x_cand
            best_y = y_cand
            best_vol = vol_cand
        
        restarts += 1
        if restarts > 150:  # Safety
            break
    
    # Final polish with smaller steps
    x_positions = best_x.copy()
    y_positions = best_y.copy()
    final_vol = best_vol
    
    for _ in range(200):
        if time.time() - start_time > max_time - 5:
            break
        
        improved = False
        for i in range(N):
            for d in [-0.005, -0.002, -0.001, 0.001, 0.002, 0.005]:
                x_new = x_positions.copy()
                x_new[i] += d
                v = compute_volume(x_new, y_positions)
                if v < final_vol:
                    x_positions = x_new
                    final_vol = v
                    improved = True
                    break
        
        for j in range(N):
            for d in [-0.005, -0.002, -0.001, 0.001, 0.002, 0.005]:
                y_new = y_positions.copy()
                y_new[j] += d
                v = compute_volume(x_positions, y_new)
                if v < final_vol:
                    y_positions = y_new
                    final_vol = v
                    improved = True
                    break
        
        if not improved:
            break
        
        best_x = x_positions.copy()
        best_y = y_positions.copy()
        best_vol = final_vol
    
    x_positions = best_x
    y_positions = best_y
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_positions[i_flat]
    base_y = y_positions[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
