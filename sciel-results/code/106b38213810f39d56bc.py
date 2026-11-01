import numpy as np
import time
from scipy.optimize import minimize

N = 50
_START_TIME = None

def _init_hex(r_guess=0.07):
    """Initialize points in a hexagonal pattern."""
    dx = 2 * r_guess
    dy = np.sqrt(3) * r_guess
    
    points = []
    # Generate a grid of hex points
    # y ranges from -0.5 to 0.5
    y = -0.5 + dy/2
    row_idx = 0
    while y <= 0.5:
        x_start = -0.5 + dx/2
        if row_idx % 2 == 1:
            x_start += dx/2
        x = x_start
        while x <= 0.5:
            points.append((x, y))
            x += dx
        y += dy
        row_idx += 1
        
    points = np.array(points)
    
    if len(points) >= N:
        # Randomly select N points to avoid bias
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]
    else:
        # Pad with random points if needed
        remaining = N - len(points)
        random_pts = np.random.uniform(-0.45, 0.45, (remaining, 2))
        return np.vstack([points, random_pts])

def _softmin_obj_and_grad(x, beta):
    x = x.reshape(N, 2)
    # Compute pair distances
    dx = x[:, 0:1] - x[:, 0:1].T
    dy = x[:, 1:2] - x[:, 1:2].T
    dists_sq = dx**2 + dy**2
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    
    # Wall distances
    wx = 0.5 - np.abs(x[:, 0])
    wy = 0.5 - np.abs(x[:, 1])
    walls = np.minimum(wx, wy)
    
    # Values for softmin: pair distances and 2*wall distances
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    wall_vals = 2 * walls
    
    vals = np.concatenate([pair_dists, wall_vals])
    
    # Softmin calculation
    # S = -1/beta * log(sum(exp(-beta * vals)))
    # To be stable:
    max_val = np.max(vals)
    exp_vals = np.exp(-beta * (vals - max_val))
    sum_exp = np.sum(exp_vals)
    # S = -1/beta * ( -beta*max_val + log(sum_exp) ) = max_val - 1/beta * log(sum_exp)
    softmin_val = max_val - (1.0 / beta) * np.log(sum_exp)
    
    # We want to maximize softmin_val.
    # So we minimize -softmin_val.
    obj_val = -softmin_val
    
    # Gradient of softmin wrt vals
    # p_k = exp(-beta * (vals_k - max_val)) / sum_exp
    p = exp_vals / sum_exp
    
    n_pairs = len(pair_dists)
    pair_p = p[:n_pairs]
    wall_p = p[n_pairs:]
    
    # Gradient of softmin wrt x
    g_soft = np.zeros_like(x)
    
    # Pairs - vectorized approach
    for idx in range(n_pairs):
        p_k = pair_p[idx]
        if p_k < 1e-15:
            continue
        i, j = iu[0][idx], iu[1][idx]
        d = pair_dists[idx]
        if d < 1e-8:
            continue
        diff = x[i] - x[j]
        # d(dist)/dx_i = diff/d
        # d(dist)/dx_j = -diff/d
        g_soft[i] += p_k * (diff / d)
        g_soft[j] -= p_k * (diff / d)
        
    # Walls
    for i in range(N):
        p_k = wall_p[i]
        if p_k < 1e-15:
            continue
        wx_i = wx[i]
        wy_i = wy[i]
        # val = 2 * min(wx, wy)
        # d(val)/dx = 2 * d(min)/dx
        if wx_i < wy_i:
            g_soft[i, 0] += p_k * 2.0 * (-np.sign(x[i, 0]))
        elif wy_i < wx_i:
            g_soft[i, 1] += p_k * 2.0 * (-np.sign(x[i, 1]))
        else:
            g_soft[i, 0] += p_k * 2.0 * (-np.sign(x[i, 0])) * 0.5
            g_soft[i, 1] += p_k * 2.0 * (-np.sign(x[i, 1])) * 0.5
            
    # Objective is -softmin, so gradient is -g_soft
    grad_obj = -g_soft
    
    return obj_val, grad_obj

def _compute_radius(x):
    """Compute the actual radius (score) for a configuration."""
    x = x.reshape(N, 2)
    dx = x[:, 0:1] - x[:, 0:1].T
    dy = x[:, 1:2] - x[:, 1:2].T
    dists_sq = dx**2 + dy**2
    np.fill_diagonal(dists_sq, 1.0)
    min_dist = np.sqrt(np.min(dists_sq))
    
    wx = 0.5 - np.abs(x[:, 0])
    wy = 0.5 - np.abs(x[:, 1])
    min_wall = np.min(np.minimum(wx, wy))
    
    return min(min_dist / 2.0, min_wall)

def construct():
    global _START_TIME
    _START_TIME = time.time()
    deadline = _START_TIME + 140.0  # Safety margin
    
    best_x = None
    best_score = -np.inf
    
    # Phase 1: Exploration with lower beta
    # Try multiple restarts with diverse initializations
    num_restarts = 25
    
    for k in range(num_restarts):
        elapsed = time.time() - _START_TIME
        if elapsed > deadline:
            break
        
        # Adaptive beta: start lower for exploration, increase gradually
        beta = 50.0 + 50.0 * (k / num_restarts)
        
        # Initialization strategy varies with k
        if k == 0:
            x0 = _init_hex(0.07)
        elif k == 1:
            x0 = _init_hex(0.065)
        elif k == 2:
            x0 = _init_hex(0.06)
        elif k % 3 == 0:
            # Grid-based init
            m = 7
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            if len(grid) >= N:
                idx = np.random.choice(len(grid), N, replace=False)
                x0 = grid[idx]
            else:
                x0 = _init_hex(0.06)
        elif k % 3 == 1:
            # Random init
            x0 = np.random.uniform(-0.45, 0.45, (N, 2))
        else:
            # Perturbed hex
            x0 = _init_hex(0.065) + np.random.normal(0, 0.01, (N, 2))
            x0 = np.clip(x0, -0.48, 0.48)
        
        # Ensure x0 is valid (finite, in bounds)
        if not np.all(np.isfinite(x0)):
            continue
        
        # Optimization
        try:
            res = minimize(
                _softmin_obj_and_grad,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            x_opt = res.x.reshape(N, 2)
            
            # Calculate exact score
            score = _compute_radius(x_opt)
            
            if score > best_score:
                best_score = score
                best_x = x_opt.copy()
                
        except Exception as e:
            pass
    
    # Phase 2: Polish - refine the best solution with higher beta and tighter tolerances
    if best_x is not None and time.time() - _START_TIME < deadline - 5.0:
        polish_beta = 200.0
        try:
            res = minimize(
                _softmin_obj_and_grad,
                best_x,
                args=(polish_beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            x_polished = res.x.reshape(N, 2)
            score_polished = _compute_radius(x_polished)
            
            if score_polished > best_score:
                best_score = score_polished
                best_x = x_polished.copy()
        except Exception as e:
            pass
    
    if best_x is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        if len(grid) < N:
             # Pad with random
             best_x = np.random.uniform(-0.4, 0.4, (N, 2))
        else:
             best_x = grid[:N]
             
    return best_x
