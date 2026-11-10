# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def _exact_1d_volume(x):
    """
    Compute the exact integral of the length of the union of sheared intervals
    [x_i + c*i/N, x_i + 1/N + c*i/N] for c in [0, 1].
    Uses closed-form trapezoidal rule on critical points.
    """
    N = 8
    x = np.asarray(x, dtype=np.float64)
    s = np.arange(N) / N
    
    # Find all critical points where any two endpoints coincide
    criticals = [0.0, 1.0]
    inv_ds = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                inv_ds[i, j] = 1.0 / (s[i] - s[j])
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            for off_i in [0.0, 1.0/N]:
                for off_j in [0.0, 1.0/N]:
                    c_val = (x[j] + off_j - x[i] - off_i) * inv_ds[i, j]
                    if 0.0 < c_val < 1.0:
                        criticals.append(c_val)
    
    # Remove duplicates and sort
    criticals = sorted(set([round(c, 15) for c in criticals]))
    
    def union_length_at_c(c_val):
        starts = x + s * c_val
        ends = starts + 1.0/N
        idx = np.argsort(starts)
        starts_sorted = starts[idx]
        ends_sorted = ends[idx]
        
        current_max_end = -np.inf
        length = 0.0
        for i in range(N):
            si = starts_sorted[i]
            ei = ends_sorted[i]
            if si > current_max_end:
                length += ei - si
                current_max_end = ei
            else:
                if ei > current_max_end:
                    length += ei - current_max_end
                    current_max_end = ei
        return length
    
    # Evaluate union length at all critical points
    n_crit = len(criticals)
    lengths = np.empty(n_crit)
    for k in range(n_crit):
        lengths[k] = union_length_at_c(criticals[k])
    
    # Trapezoidal rule: integral = sum over segments of 0.5*(L_k + L_{k+1})*dc
    dc = np.diff(criticals)
    avg_lengths = 0.5 * (lengths[:-1] + lengths[1:])
    total_vol = np.sum(avg_lengths * dc)
    
    return total_vol

def _objective(params):
    """
    Objective function: V = Vol_x * Vol_y
    """
    x = params[:8]
    y = params[8:16]
    vx = _exact_1d_volume(x)
    vy = _exact_1d_volume(y)
    return vx * vy

def _objective_1d(x, fixed_other_vol):
    """
    Objective for optimizing one dimension given the other is fixed.
    """
    vx = _exact_1d_volume(x)
    return vx * fixed_other_vol

def kakeya_needle_3d8() -> np.ndarray:
    """
    Optimize the base positions to minimize the union volume.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Initial guess: Keich construction from seed
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
    
    bounds = [(-1.0, 1.0)] * 16
    bounds_1d = [(-1.0, 1.0)] * 8
    
    best_params = params0.copy()
    best_score = _objective(params0)
    
    # Phase 1: Differential Evolution for global search
    try:
        res = scipy.optimize.differential_evolution(
            _objective, 
            bounds, 
            maxiter=300, 
            popsize=12, 
            seed=42, 
            tol=1e-10,
            polish=True,
            polish_options={'maxiter': 2000, 'ftol': 1e-14}
        )
        
        if res.fun < best_score:
            best_params = res.x.copy()
            best_score = res.fun
    except Exception:
        pass
    
    # Phase 2: Per-dimension local optimization (8D sub-problems)
    if time.time() < deadline - 120:
        x_opt = best_params[:8].copy()
        y_opt = best_params[8:16].copy()
        
        # Optimize x with y fixed
        vy_fixed = _exact_1d_volume(y_opt)
        try:
            res_x = scipy.optimize.minimize(
                lambda x: _exact_1d_volume(x) * vy_fixed,
                x_opt,
                method='L-BFGS-B',
                bounds=bounds_1d,
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if res_x.fun < _exact_1d_volume(x_opt) * vy_fixed:
                x_opt = res_x.x.copy()
        except Exception:
            pass
        
        # Optimize y with x fixed
        vx_fixed = _exact_1d_volume(x_opt)
        try:
            res_y = scipy.optimize.minimize(
                lambda y: _exact_1d_volume(y) * vx_fixed,
                y_opt,
                method='L-BFGS-B',
                bounds=bounds_1d,
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if res_y.fun < _exact_1d_volume(y_opt) * vx_fixed:
                y_opt = res_y.x.copy()
        except Exception:
            pass
        
        # Iterate a few times
        for _ in range(3):
            if time.time() > deadline - 100:
                break
            vy_fixed = _exact_1d_volume(y_opt)
            try:
                res_x = scipy.optimize.minimize(
                    lambda x: _exact_1d_volume(x) * vy_fixed,
                    x_opt,
                    method='L-BFGS-B',
                    bounds=bounds_1d,
                    options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                if res_x.fun < _exact_1d_volume(x_opt) * vy_fixed:
                    x_opt = res_x.x.copy()
            except Exception:
                pass
            
            vx_fixed = _exact_1d_volume(x_opt)
            try:
                res_y = scipy.optimize.minimize(
                    lambda y: _exact_1d_volume(y) * vx_fixed,
                    y_opt,
                    method='L-BFGS-B',
                    bounds=bounds_1d,
                    options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                if res_y.fun < _exact_1d_volume(y_opt) * vx_fixed:
                    y_opt = res_y.x.copy()
            except Exception:
                pass
        
        candidate = np.concatenate([x_opt, y_opt])
        cand_score = _objective(candidate)
        if cand_score < best_score:
            best_params = candidate
            best_score = cand_score
    
    # Phase 3: Final joint polish
    if time.time() < deadline - 60:
        try:
            res_final = scipy.optimize.minimize(
                _objective, 
                best_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            if res_final.fun < best_score:
                best_params = res_final.x.copy()
                best_score = res_final.fun
        except Exception:
            pass
    
    # Construct the output array
    x_opt = best_params[:8]
    y_opt = best_params[8:16]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
