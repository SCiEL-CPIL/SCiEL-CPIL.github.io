import numpy as np
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    deadline = start_time + 620

    def compute_volume(x_rows, y_cols, rng, n_samples=500000):
        """
        x_rows: (8,) array of x_i for each row i
        y_cols: (8,) array of y_j for each column j
        Tube (i,j) at height c: x in [x_i + c*i/8, x_i + c*i/8 + 1/8], y in [y_j + c*j/8, y_j + c*j/8 + 1/8]
        """
        if not np.all(np.isfinite(x_rows)) or not np.all(np.isfinite(y_cols)):
            return np.inf
        
        i_arr = np.arange(N)
        j_arr = np.arange(N)
        
        # Box bounds
        # x range for tube (i,j): [x_i + c*i/8, x_i + c*i/8 + 1/8] for c in [0,1]
        # So x range: [min_i x_i, max_i (x_i + i/8 + 1/8)]
        X0 = np.min(x_rows)
        X1 = np.max(x_rows + (i_arr + 1) / N)
        Y0 = np.min(y_cols)
        Y1 = np.max(y_cols + (j_arr + 1) / N)
        
        if X1 - X0 <= 1e-9 or Y1 - Y0 <= 1e-9:
            return np.inf
        
        a = rng.uniform(X0, X1, n_samples)
        b = rng.uniform(Y0, Y1, n_samples)
        c = rng.uniform(0, 1, n_samples)
        
        # For each sample point (a, b, c), check if inside any tube
        # Tube (i,j): a in [x_i + c*i/N, x_i + c*i/N + 1/N] AND b in [y_j + c*j/N, y_j + c*j/N + 1/N]
        
        # Vectorized: for each (i,j), compute whether point is inside
        # x_lo[i] = x_i + c*i/N, x_hi[i] = x_lo[i] + 1/N
        # y_lo[j] = y_j + c*j/N, y_hi[j] = y_lo[j] + 1/N
        
        x_lo = x_rows[:, None] + c[None, :] * i_arr[:, None] / N  # (8, n_samples)
        x_hi = x_lo + 1.0 / N
        y_lo = y_cols[:, None] + c[None, :] * j_arr[:, None] / N  # (8, n_samples)
        y_hi = y_lo + 1.0 / N
        
        in_x = (a[None, :] >= x_lo) & (a[None, :] <= x_hi)  # (8, n_samples)
        in_y = (b[None, :] >= y_lo) & (b[None, :] <= y_hi)  # (8, n_samples)
        
        # Tube (i,j) contains point if in_x[i] AND in_y[j]
        # We need OR over all (i,j)
        # inside[i,j] = in_x[i] & in_y[j]
        # inside_any = any over i,j
        
        # This is: for each sample, is there any i,j such that in_x[i] & in_y[j]?
        # = (any i: in_x[i]) & (any j: in_y[j])? No, that's not right.
        # Actually: exists (i,j) such that in_x[i] AND in_y[j]
        # This is equivalent to: (exists i: in_x[i]) AND (exists j: in_y[j])
        # Because if there's some i with in_x[i] true and some j with in_y[j] true,
        # then tube (i,j) contains the point.
        
        any_in_x = np.any(in_x, axis=0)  # (n_samples,)
        any_in_y = np.any(in_y, axis=0)  # (n_samples,)
        inside_any = any_in_x & any_in_y
        
        fraction = np.mean(inside_any)
        V = fraction * (X1 - X0) * (Y1 - Y0)
        return V

    # Keich-inspired initialization
    # For the separable structure, we want x_i and y_j to create maximum overlap
    # A good heuristic: spread the base positions to minimize the bounding box while allowing overlap
    
    # Try several initialization strategies
    best_V = np.inf
    best_x = None
    best_y = None
    
    rng_init = np.random.default_rng(42)
    
    # Strategy 1: Keich-style binary offsets
    def keich_offsets(N, pos):
        j_terms = np.arange(1, pos + 1)
        ks = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            ks[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        return ks
    
    pos = int(np.log2(N))  # 3
    keich_base = keich_offsets(N, pos)
    
    # Strategy 2: Uniform spacing
    uniform_base = np.linspace(0, 0.5, N)
    
    # Strategy 3: Random
    random_base = rng_init.uniform(0, 0.5, N)
    
    init_strategies = [keich_base, uniform_base, random_base]
    
    for strat_idx, base in enumerate(init_strategies):
        if time.time() > deadline - 30:
            break
        
        x_init = base.copy()
        y_init = base.copy()
        
        # Evaluate initial
        rng_eval = np.random.default_rng(12345)
        V = compute_volume(x_init, y_init, rng_eval)
        
        if V < best_V and np.isfinite(V):
            best_V = V
            best_x = x_init.copy()
            best_y = y_init.copy()
        
        # Local optimization using Nelder-Mead on the 16D parameter space
        if time.time() > deadline - 60:
            break
        
        from scipy.optimize import minimize
        
        def objective(params):
            x = params[:N]
            y = params[N:]
            rng_obj = np.random.default_rng(12345)
            return compute_volume(x, y, rng_obj, n_samples=100000)
        
        try:
            result = minimize(
                objective,
                np.concatenate([x_init, y_init]),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-5, 'adaptive': True}
            )
            
            if time.time() < deadline - 30:
                x_opt = result.x[:N]
                y_opt = result.x[N:]
                if np.all(np.isfinite(result.x)) and result.fun < best_V:
                    best_V = result.fun
                    best_x = x_opt.copy()
                    best_y = y_opt.copy()
        except Exception:
            pass
    
    # Additional refinement with more samples
    if time.time() < deadline - 30 and best_x is not None:
        rng_final = np.random.default_rng(12345)
        V_final = compute_volume(best_x, best_y, rng_final, n_samples=500000)
        if V_final < best_V:
            best_V = V_final
    
    # Try a few more random restarts
    rng_restart = np.random.default_rng(777)
    for restart in range(3):
        if time.time() > deadline - 40:
            break
        
        noise = 0.02
        x_try = best_x + rng_restart.normal(0, noise, N)
        y_try = best_y + rng_restart.normal(0, noise, N)
        
        rng_eval = np.random.default_rng(12345)
        V_try = compute_volume(x_try, y_try, rng_eval, n_samples=100000)
        
        if V_try < best_V and np.isfinite(V_try):
            best_V = V_try
            best_x = x_try.copy()
            best_y = y_try.copy()
    
    # Build the full 64x2 position array
    if best_x is None or best_y is None:
        # Fallback
        best_x = keich_base
        best_y = keich_base
    
    i_vals = np.arange(N)
    j_vals = np.arange(N)
    i_grid, j_grid = np.meshgrid(i_vals, j_vals, indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    bx = best_x[i_flat]
    by = best_y[j_flat]
    pos_final = np.stack([bx, by], axis=1)
    
    assert pos_final.shape == (64, 2)
    assert np.all(np.isfinite(pos_final))
    
    return pos_final
