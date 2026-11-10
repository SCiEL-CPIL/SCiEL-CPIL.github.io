import numpy as np
from scipy.optimize import minimize
import time

def _union_len_1d(positions, c, N=8):
    """Compute union length of N intervals [p_i + c*i/N, p_i + c*i/N + 1/N]."""
    frac = 1.0 / N
    starts = positions + c * np.arange(N) * frac
    ends = starts + frac
    # Sort by start
    order = np.argsort(starts)
    s = starts[order]
    e = ends[order]
    # Sweep: union length
    total = 0.0
    cur_start = s[0]
    cur_end = e[0]
    for i in range(1, N):
        if s[i] > cur_end:
            total += cur_end - cur_start
            cur_start = s[i]
            cur_end = e[i]
        else:
            cur_end = max(cur_end, e[i])
    total += cur_end - cur_start
    return total

def _soft_union_len_1d(positions, c, N=8, beta=30.0):
    """Smooth approximation of union length using soft-max of coverage."""
    frac = 1.0 / N
    starts = positions + c * np.arange(N) * frac
    ends = starts + frac
    
    # All breakpoints
    all_pts = np.sort(np.concatenate([starts, ends]))
    n_pts = len(all_pts)
    
    # Cell centers and widths
    cx = 0.5 * (all_pts[:-1] + all_pts[1:])
    dx = np.diff(all_pts)
    
    # For each cell center, compute soft coverage by each interval
    # dist[i,k] = min(cx_k - starts_i, ends_i - cx_k)  (signed distance to interval)
    left = cx[None, :] - starts[:, None]
    right = ends[:, None] - cx[None, :]
    dist = np.minimum(left, right)  # (N, n_cells)
    
    # Soft coverage: sigmoid(beta * dist)
    bz = np.clip(beta * dist, -50, 50)
    soft_cov = 1.0 / (1.0 + np.exp(-bz))  # (N, n_cells)
    
    # Soft union: for each cell, soft_max over intervals
    # soft_max = log(sum(exp(log(soft_cov))))... but soft_cov is in [0,1]
    # Better: use log-sum-exp on log(soft_cov + eps)
    eps = 1e-15
    log_soft = np.log(soft_cov + eps)
    m = np.max(log_soft, axis=0, keepdims=True)
    log_sum_exp = m + np.log(np.sum(np.exp(log_soft - m), axis=0, keepdims=True))
    soft_union_per_cell = np.exp(log_sum_exp)[0]  # (n_cells,)
    # Clip to [0,1]
    soft_union_per_cell = np.clip(soft_union_per_cell, 0, 1)
    
    area = np.sum(soft_union_per_cell * dx)
    return area

def _cost_smooth(params, c_samples, beta=30.0):
    """Smooth cost: average of x_union_len * y_union_len over c samples."""
    x = params[:8]
    y = params[8:]
    total = 0.0
    for c in c_samples:
        lx = _soft_union_len_1d(x, c, beta=beta)
        ly = _soft_union_len_1d(y, c, beta=beta)
        total += lx * ly
    return total / len(c_samples)

def _cost_exact(params, c_samples):
    """Exact cost: average of x_union_len * y_union_len over c samples."""
    x = params[:8]
    y = params[8:]
    total = 0.0
    for c in c_samples:
        lx = _union_len_1d(x, c)
        ly = _union_len_1d(y, c)
        total += lx * ly
    return total / len(c_samples)

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # safety margin
    
    N = 8
    c_samples = np.linspace(0, 1, 50)
    
    best_params = None
    best_cost = np.inf
    
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Different initializations
        if restart == 0:
            x0 = np.linspace(0, 1 - 1/N, N)
            y0 = np.linspace(0, 1 - 1/N, N)
        elif restart == 1:
            np.random.seed(42 + restart)
            x0 = np.random.uniform(0, 1 - 1/N, N)
            y0 = np.random.uniform(0, 1 - 1/N, N)
        elif restart == 2:
            x0 = np.random.uniform(0.2, 0.8, N)
            y0 = np.random.uniform(0.2, 0.8, N)
        elif restart == 3:
            np.random.seed(200 + restart)
            x0 = np.random.uniform(0, 1, N)
            y0 = np.random.uniform(0, 1, N)
        elif restart == 4:
            # Anti-grid: shift odd indices
            x0 = np.linspace(0, 1 - 1/N, N)
            x0[1::2] += 0.3
            y0 = np.linspace(0, 1 - 1/N, N)
            y0[1::2] += 0.3
        elif restart == 5:
            # Concentrated with random offset
            np.random.seed(500 + restart)
            cx = np.random.uniform(0.2, 0.6)
            cy = np.random.uniform(0.2, 0.6)
            x0 = cx + np.random.uniform(-0.15, 0.15, N)
            y0 = cy + np.random.uniform(-0.15, 0.15, N)
        elif restart == 6:
            # Diagonal spread
            x0 = np.linspace(0.1, 0.9, N)
            y0 = np.linspace(0.9, 0.1, N)
        else:
            np.random.seed(900 + restart)
            x0 = np.random.uniform(0, 1, N)
            y0 = np.random.uniform(0, 1, N)
        
        params0 = np.concatenate([x0, y0])
        cost0 = _cost_exact(params0, c_samples)
        
        if cost0 < best_cost:
            best_cost = cost0
            best_params = params0.copy()
        
        # Phase 1: L-BFGS-B on smooth surrogate
        try:
            res = minimize(_cost_smooth, params0, args=(c_samples, 30.0),
                          method='L-BFGS-B', maxiter=500,
                          options={'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 50})
            if res.fun < best_cost:
                best_cost = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        if time.time() > deadline:
            break
        
        # Phase 2: Powell on exact cost for polish
        if best_params is not None:
            try:
                res2 = minimize(_cost_exact, best_params, args=(c_samples,),
                               method='Powell', maxiter=300,
                               options={'xtol': 1e-8, 'ftol': 1e-10})
                if res2.fun < best_cost:
                    best_cost = res2.fun
                    best_params = res2.x.copy()
            except Exception:
                pass
        
        if time.time() > deadline:
            break
    
    if best_params is None:
        x0 = np.linspace(0, 1 - 1/N, N)
        y0 = np.linspace(0, 1 - 1/N, N)
        best_params = np.concatenate([x0, y0])
    
    x_opt = best_params[:8]
    y_opt = best_params[8:]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions
