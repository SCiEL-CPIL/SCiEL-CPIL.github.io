# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_3d_volume_exact(x, N=8, n_c=40):
    """
    Exact 3D union volume for the Kakeya construction.
    x: shape (N,) - base positions for each i (x_k = x[i], y_k = x[j])
    
    The 3D volume = integral over c in [0,1] of 2D union area of 64 rectangles.
    Each rectangle k=(i,j) at height c: [x_i + c*i/N, x_i + c*i/N + 1/N] x [x_j + c*j/N, x_j + c*j/N + 1/N]
    
    2D union area is computed via coordinate compression (sweep line) at each c.
    """
    N = int(N)
    n_c = int(n_c)
    c_vals = np.linspace(0.0, 1.0, n_c)
    i_idx = np.arange(N)
    
    total_area = 0.0
    for c in c_vals:
        # 64 rectangles: for each (i, j)
        # x-range: [x_i + c*i/N, x_i + c*i/N + 1/N]
        # y-range: [x_j + c*j/N, x_j + c*j/N + 1/N]
        
        # Collect all x-coordinates and y-coordinates
        x_starts = x + c * i_idx / N
        x_ends = x_starts + 1.0 / N
        y_starts = x + c * i_idx / N  # same x array used for y (j index)
        y_ends = y_starts + 1.0 / N
        
        # All unique x-coordinates
        all_x = np.concatenate([x_starts, x_ends])
        all_x = np.unique(all_x)
        # All unique y-coordinates
        all_y = np.concatenate([y_starts, y_ends])
        all_y = np.unique(all_y)
        
        if len(all_x) < 2 or len(all_y) < 2:
            total_area += 0.0
            continue
        
        # Create grid of cells
        # Cell (a, b) is [all_x[a], all_x[a+1]] x [all_y[b], all_y[b+1]]
        # Check if cell center is inside any rectangle
        
        # For efficiency, use the fact that we have only 8x8 = 64 rectangles
        # Cell center x: (all_x[a] + all_x[a+1]) / 2
        # Cell center y: (all_y[b] + all_y[b+1]) / 2
        
        cx = (all_x[:-1] + all_x[1:]) / 2.0  # shape (nx-1,)
        cy = (all_y[:-1] + all_y[1:]) / 2.0  # shape (ny-1,)
        
        # For each cell, check if covered by any of the 64 rectangles
        # Rectangle (i,j): x in [x_starts[i], x_ends[i]], y in [y_starts[j], y_ends[j]]
        
        # Vectorized: for each cell center (cx[a], cy[b]), check all 64 rectangles
        # covered[a, b] = any over (i,j) of (cx[a] >= x_starts[i] and cx[a] <= x_ends[i] and cy[b] >= y_starts[j] and cy[b] <= y_ends[j])
        
        # This is separable in x and y!
        # x_covered[a, i] = cx[a] in [x_starts[i], x_ends[i]]
        # y_covered[b, j] = cy[b] in [y_starts[j], y_ends[j]]
        # cell_covered[a, b] = any(i,j) of x_covered[a,i] AND y_covered[b,j]
        # = (exists i: x_covered[a,i]) AND (exists j: y_covered[b,j])
        # Wait, that's not right. We need: exists (i,j) such that x_covered[a,i] AND y_covered[b,j]
        # This equals: (exists i: x_covered[a,i]) AND (exists j: y_covered[b,j])
        # Because the condition factors: the x-condition only depends on i, y-condition only on j.
        # So cell (a,b) is covered iff (some i covers x) AND (some j covers y).
        
        x_cov = (cx[:, None] >= x_starts[None, :]) & (cx[:, None] <= x_ends[None, :])  # (nx-1, N)
        y_cov = (cy[:, None] >= y_starts[None, :]) & (cy[:, None] <= y_ends[None, :])  # (ny-1, N)
        
        x_any = x_cov.any(axis=1)  # (nx-1,)
        y_any = y_cov.any(axis=1)  # (ny-1,)
        
        # Cell (a,b) covered iff x_any[a] AND y_any[b]
        cell_cov = x_any[:, None] & y_any[None, :]  # (nx-1, ny-1)
        
        # Area of each cell
        dx = np.diff(all_x)  # (nx-1,)
        dy = np.diff(all_y)  # (ny-1,)
        cell_area = dx[:, None] * dy[None, :]  # (nx-1, ny-1)
        
        total_area += np.sum(cell_area * cell_cov)
    
    return total_area / n_c

def _compute_3d_volume_exact_fast(x, N=8, n_c=60):
    """
    Faster version using the separability insight.
    At each c, the 2D union area = area_x(c) * area_y(c) where
    area_x = 1D union length of intervals [x_i + c*i/N, x_i + c*i/N + 1/N]
    area_y = same (since y uses same x array)
    
    Wait - this is only true if the coverage condition factors. Let me verify:
    Cell (a,b) is covered iff exists (i,j) with cx in rect_i_x AND cy in rect_j_y.
    = (exists i: cx in rect_i_x) AND (exists j: cy in rect_j_y)
    = x_any[a] AND y_any[b]
    
    So the 2D union area = (sum over x-cells that are x-covered) * (sum over y-cells that are y-covered)
    = area_x_union * area_y_union
    
    Since x and y use the SAME x array, area_x_union = area_y_union = 1D union length.
    So 2D area = (1D union length)^2!
    """
    N = int(N)
    n_c = int(n_c)
    c_vals = np.linspace(0.0, 1.0, n_c)
    i_idx = np.arange(N)
    
    total = 0.0
    for c in c_vals:
        starts = x + c * i_idx / N
        ends = starts + 1.0 / N
        
        # 1D union length
        order = np.argsort(starts)
        s = starts[order]
        e = ends[order]
        
        union_len = 0.0
        cur_s = s[0]
        cur_e = e[0]
        for i in range(1, N):
            if s[i] > cur_e:
                union_len += cur_e - cur_s
                cur_s = s[i]
                cur_e = e[i]
            else:
                if e[i] > cur_e:
                    cur_e = e[i]
        union_len += cur_e - cur_s
        
        # 2D area = union_len^2 (by separability)
        total += union_len * union_len
    
    return total / n_c

def _compute_3d_volume_smooth(x, N=8, n_c=40, beta=50.0):
    """
    Smooth surrogate for the 3D volume.
    Uses the fact that 2D area = (1D union length)^2.
    For the 1D union length, use a smooth approximation of the max/min operations.
    
    Actually, let's use a different approach: the 1D union length can be computed
    as sum of positive parts. For sorted intervals, union = sum of (e_i - max(s_i, prev_e))_+.
    We can smooth the max with a soft-max.
    """
    N = int(N)
    n_c = int(n_c)
    c_vals = np.linspace(0.0, 1.0, n_c)
    i_idx = np.arange(N)
    
    total = 0.0
    for c in c_vals:
        starts = x + c * i_idx / N
        ends = starts + 1.0 / N
        
        order = np.argsort(starts)
        s = starts[order]
        e = ends[order]
        
        # Smooth union: union = s[0] to e[N-1] minus gaps
        # Gap at i: max(0, s[i] - e[i-1]) for i=1..N-1
        # Smooth max(0, z) = logsumexp(0, z) / beta * beta = softplus-like
        # Actually: softplus(z) = log(1 + exp(z)) approximates max(0, z) for large beta
        # Use: smooth_max0(z) = (1/beta) * log(1 + exp(beta*z))
        
        gaps = s[1:] - e[:-1]  # (N-1,)
        # smooth_max0(gap) = (1/beta)*log(1 + exp(beta*gap))
        beta_val = beta
        # Clip for numerical stability
        gaps_clipped = np.clip(gaps, -50.0/beta_val, 50.0/beta_val)
        smooth_gaps = (1.0 / beta_val) * np.log1p(np.exp(beta_val * gaps_clipped))
        
        total_gap = np.sum(smooth_gaps)
        
        # Total span
        span = e[-1] - s[0]
        union_len = span - total_gap
        
        if union_len < 0:
            union_len = 0.0
        
        total += union_len * union_len
    
    return total / n_c

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    t_start = time.time()
    deadline = t_start + 590.0  # 10s safety margin
    
    # Initial candidates
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    
    # Keich construction
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # Symmetric
    symmetric = -np.arange(N) / (2 * N)
    
    # Origin
    origin = np.zeros(N)
    
    # Linear
    linear = np.linspace(0, 1 - 1/N, N)
    
    # Shifted symmetric
    shifted_sym = symmetric + 0.5/(2*N)
    
    candidates = [keich, symmetric, origin, linear, shifted_sym]
    
    best_x = keich.copy()
    best_vol = float('inf')
    
    for x in candidates:
        vol = _compute_3d_volume_exact_fast(x, N, 200)
        if vol < best_vol:
            best_vol = vol
            best_x = x.copy()
    
    # Local optimization using the smooth surrogate for gradient
    if time.time() < deadline - 300.0:
        from scipy.optimize import minimize
        
        def objective(x):
            return _compute_3d_volume_smooth(x, N, 40, beta=80.0)
        
        def objective_grad(x):
            # Numerical gradient
            eps = 1e-7
            g = np.zeros_like(x)
            for i in range(N):
                xp = x.copy()
                xm = x.copy()
                xp[i] += eps
                xm[i] -= eps
                g[i] = (objective(xp) - objective(xm)) / (2 * eps)
            return g
        
        x0 = best_x.copy()
        try:
            res = minimize(objective, x0, method='L-BFGS-B', jac=objective_grad,
                          options={'maxiter': 300, 'maxls': 30, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_vol:
                best_vol = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
        
        # Nelder-Mead for non-smooth objective
        if time.time() < deadline - 150.0:
            def objective_ns(x):
                return _compute_3d_volume_exact_fast(x, N, 100)
            
            try:
                res2 = minimize(objective_ns, best_x, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12,
                                       'adaptive': True})
                if res2.fun < best_vol:
                    best_vol = res2.fun
                    best_x = res2.x.copy()
            except Exception:
                pass
    
    # Multi-start refinement
    if time.time() < deadline - 120.0:
        from scipy.optimize import minimize
        
        def objective_fast(x):
            return _compute_3d_volume_exact_fast(x, N, 80)
        
        # Try several random perturbations
        rng = np.random.default_rng(42)
        for trial in range(5):
            if time.time() > deadline - 60.0:
                break
            
            perturbation = rng.normal(0, 0.05, N)
            x0 = best_x + perturbation
            
            try:
                res = minimize(objective_fast, x0, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                                       'adaptive': True})
                if res.fun < best_vol:
                    best_vol = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
    
    # Final high-precision evaluation
    best_vol = _compute_3d_volume_exact_fast(best_x, N, 1000)
    
    # Build the 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_x[i_flat]
    base_y = best_x[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
