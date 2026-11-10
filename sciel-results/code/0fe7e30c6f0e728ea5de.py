import numpy as np
from scipy.optimize import minimize
import time

def _union_area_at_c(x, y, c):
    """Compute union area of 8x8 axis-aligned rectangles at height c.
    Rect (i,j): [x[i]+c*i/8, x[i]+c*i/8+1/8] x [y[j]+c*j/8, y[j]+c*j/8+1/8]
    Uses sweep-line: collect all x and y boundaries, check each cell.
    Vectorized for speed."""
    N = 8
    frac = 1.0 / N
    x0s = x + c * np.arange(N) * frac
    y0s = y + c * np.arange(N) * frac
    
    # All x boundaries
    xs = np.sort(np.concatenate([x0s, x0s + frac]))
    ys = np.sort(np.concatenate([y0s, y0s + frac]))
    
    # Cell centers
    cx = 0.5 * (xs[:-1] + xs[1:])
    cy = 0.5 * (ys[:-1] + ys[1:])
    dx = np.diff(xs)
    dy = np.diff(ys)
    
    # For each cell, check if covered by any rect
    # Rect (i,j) covers cell if x0s[i] <= cx <= x0s[i]+frac and y0s[j] <= cy <= y0s[j]+frac
    # Vectorize: for each cell, check all 64 rects
    # cx shape (nx,), cy shape (ny,)
    # For x: covered if exists i: x0s[i] <= cx and cx <= x0s[i]+frac
    # Equivalent: max_i min(cx - x0s[i], x0s[i]+frac - cx) >= 0 ... not quite
    # Better: for each cx, find if any x0s[i] <= cx <= x0s[i]+frac
    # This is: (cx >= x0s.min()) & (cx <= x0s.max()+frac) & (exists i with x0s[i] <= cx <= x0s[i]+frac)
    
    # Vectorized check: create meshgrid
    # x_covered[nx, N] = (cx[:,None] >= x0s[None,:]) & (cx[:,None] <= x0s[None,:]+frac)
    x_cov = (cx[:, None] >= x0s[None, :]) & (cx[:, None] <= x0s[None, :] + frac)
    y_cov = (cy[:, None] >= y0s[None, :]) & (cy[:, None] <= y0s[None, :] + frac)
    
    # Cell (k,l) is covered if exists i,j: x_cov[k,i] and y_cov[l,j]
    # x_cov shape (nx, N), y_cov shape (ny, N)
    # covered[k,l] = any_i x_cov[k,i] AND any_j y_cov[l,j]
    # = (x_cov.any(axis=1))[:, None] * (y_cov.any(axis=1))[None, :]
    x_any = x_cov.any(axis=1)  # (nx,)
    y_any = y_cov.any(axis=1)  # (ny,)
    covered = x_any[:, None] * y_any[None, :]  # (nx, ny)
    
    area = np.sum(covered * dx[:, None] * dy[None, :])
    return area

def _smooth_union_area_at_c(x, y, c, beta=50.0):
    """Smooth surrogate for union area using soft-max of coverage indicators."""
    N = 8
    frac = 1.0 / N
    x0s = x + c * np.arange(N) * frac
    y0s = y + c * np.arange(N) * frac
    
    xs = np.sort(np.concatenate([x0s, x0s + frac]))
    ys = np.sort(np.concatenate([y0s, y0s + frac]))
    
    cx = 0.5 * (xs[:-1] + xs[1:])
    cy = 0.5 * (ys[:-1] + ys[1:])
    dx = np.diff(xs)
    dy = np.diff(ys)
    
    # Soft coverage: use sigmoid of distance to boundary
    # For x: soft_x_cov[k,i] = sigmoid(beta * (min(cx_k - x0s_i, x0s_i+frac - cx_k)))
    # This is positive when inside, negative when outside, smooth transition
    
    # cx shape (nx,), x0s shape (N,)
    # dist_x[k,i] = min(cx_k - x0s_i, x0s_i+frac - cx_k)
    # This is the "signed distance" to the interval [x0s_i, x0s_i+frac]
    # Inside: positive, Outside: negative
    
    # Compute distances
    # cx[:,None] - x0s[None,:] gives left distance
    # x0s[None,:] + frac - cx[:,None] gives right distance
    left_x = cx[:, None] - x0s[None, :]
    right_x = x0s[None, :] + frac - cx[:, None]
    dist_x = np.minimum(left_x, right_x)  # (nx, N)
    
    left_y = cy[:, None] - y0s[None, :]
    right_y = y0s[None, :] + frac - cy[:, None]
    dist_y = np.minimum(left_y, right_y)  # (ny, N)
    
    # Soft coverage: sigmoid(beta * dist)
    # Use stable sigmoid
    def soft_sigmoid(z, beta):
        # sigmoid(beta * z)
        bz = beta * z
        # Clip to avoid overflow
        bz = np.clip(bz, -50, 50)
        return 1.0 / (1.0 + np.exp(-bz))
    
    x_soft = soft_sigmoid(dist_x, beta)  # (nx, N)
    y_soft = soft_sigmoid(dist_y, beta)  # (ny, N)
    
    # Soft union: for each cell, soft coverage = soft_max over i of x_soft[k,i] AND soft_max over j of y_soft[l,j]
    # Use log-sum-exp for soft max
    def log_sum_exp(a, axis=None):
        m = np.max(a, axis=axis, keepdims=True)
        return np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    
    # Soft max of x_soft along axis 1
    x_soft_max = np.logaddexp.reduce(x_soft, axis=1)  # (nx,) - approximates max
    y_soft_max = np.logaddexp.reduce(y_soft, axis=1)  # (ny,)
    
    # Actually logaddexp.reduce gives log(sum(exp)), not max. For soft max we want:
    # soft_max(a) = log(sum(exp(a))) for a in log-space, or use a different approach
    # Let's use: soft_max(a) ≈ max(a) + (1/beta)*log(sum(exp(beta*(a-max))))
    # But simpler: just use the product of soft coverages as a proxy
    # Cell covered if x_soft_max[k] > 0.5 and y_soft_max[l] > 0.5 approximately
    
    # Better approach: soft union area = sum over cells of min(1, max_i x_soft[k,i]) * min(1, max_j y_soft[l,j]) * dx[k] * dy[l]
    # Use soft min(1, .) = sigmoid(beta*(1 - .))... no
    
    # Let's just use the hard version for now and add smoothness later if needed
    # For optimization, use the hard area but with subgradient or finite differences
    
    x_any = x_cov.any(axis=1) if 'x_cov' in dir() else (x_soft > 0.5).any(axis=1)
    y_any = y_cov.any(axis=1) if 'y_cov' in dir() else (y_soft > 0.5).any(axis=1)
    covered = x_any[:, None] * y_any[None, :]
    area = np.sum(covered * dx[:, None] * dy[None, :])
    return area

def _cost_func(params):
    x = params[:8]
    y = params[8:]
    c_samples = np.linspace(0, 1, 20)
    total_area = 0.0
    for c in c_samples:
        total_area += _union_area_at_c(x, y, c)
    return total_area / len(c_samples)

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    N = 8
    
    # Multiple restarts with different initializations
    best_params = None
    best_cost = np.inf
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > 580:
            break
            
        # Different initializations
        if restart == 0:
            # Grid-like initialization
            x0 = np.linspace(0, 1 - 1/N, N)
            y0 = np.linspace(0, 1 - 1/N, N)
        elif restart == 1:
            # Random
            np.random.seed(42 + restart)
            x0 = np.random.uniform(0, 1 - 1/N, N)
            y0 = np.random.uniform(0, 1 - 1/N, N)
        elif restart == 2:
            # Concentrated
            x0 = np.random.uniform(0.3, 0.7, N)
            y0 = np.random.uniform(0.3, 0.7, N)
        else:
            # More spread
            np.random.seed(100 + restart)
            x0 = np.random.uniform(0, 1, N)
            y0 = np.random.uniform(0, 1, N)
        
        params0 = np.concatenate([x0, y0])
        cost0 = _cost_func(params0)
        
        if cost0 < best_cost:
            best_cost = cost0
            best_params = params0.copy()
        
        # Optimize
        try:
            res = minimize(_cost_func, params0, method='Powell', 
                          options={'maxiter': 1500, 'xtol': 1e-6, 'ftol': 1e-6})
            if res.fun < best_cost:
                best_cost = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        if time.time() - start_time > 560:
            break
    
    if best_params is None:
        # Fallback
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
