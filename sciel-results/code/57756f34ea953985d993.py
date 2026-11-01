import numpy as np
import time
from scipy.optimize import minimize

N = 50

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
    if len(points) < N:
        # If not enough, add random points or expand grid
        # This shouldn't happen for r=0.07 in unit square
        # Area ~ 1. Hex density ~ 1/(2*r^2 * sqrt(3)/2) = 1/(r^2 sqrt(3))
        # r=0.07 -> r^2=0.0049. 1/(0.0049*1.73) ~ 118 points.
        # So we have plenty.
        pass
        
    # If we have more than N, select N closest to center?
    # Or just take first N?
    # Taking first N from a hex grid starting at -0.5 might be biased.
    # Let's select N points that are most "spread out" or just random subset?
    # Actually, for packing, we want to fill the space.
    # Let's just take the first N points from the generated list, but ensure we cover the square.
    # A better init: Sort by distance to center, take N? No, that clusters in center.
    # Let's just take a random subset of the hex grid points?
    # Or, simpler: Generate a grid, and if count > N, randomly drop N-count points?
    # Let's just use the first N. The optimizer will move them.
    
    if len(points) >= N:
        # Randomly select N points to avoid bias?
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]
    else:
        return points

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
    
    # Pairs
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_x = None
    best_score = -np.inf
    
    # Try multiple restarts
    num_restarts = 20
    
    for k in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialization
        if k == 0:
            x0 = _init_hex(0.07)
        elif k % 2 == 1:
            x0 = _init_hex(0.06) # Sparser start
        else:
            # Random init with some repulsion?
            # Just random in [-0.45, 0.45]
            x0 = np.random.uniform(-0.45, 0.45, (N, 2))
            
        # Ensure x0 is valid (finite, in bounds)
        if not np.all(np.isfinite(x0)):
            continue
            
        # Optimization
        # Use L-BFGS-B
        # beta: Start with 50, maybe increase?
        # Let's try a fixed beta=100 for simplicity and speed.
        beta = 100.0
        
        try:
            res = minimize(
                _softmin_obj_and_grad,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            x_opt = res.x.reshape(N, 2)
            
            # Calculate exact score
            # r = min(min_dist/2, min_wall)
            dx = x_opt[:, 0:1] - x_opt[:, 0:1].T
            dy = x_opt[:, 1:2] - x_opt[:, 1:2].T
            dists_sq = dx**2 + dy**2
            np.fill_diagonal(dists_sq, 1.0)
            min_dist = np.sqrt(np.min(dists_sq))
            
            wx = 0.5 - np.abs(x_opt[:, 0])
            wy = 0.5 - np.abs(x_opt[:, 1])
            min_wall = np.min(np.minimum(wx, wy))
            
            score = min(min_dist / 2.0, min_wall)
            
            if score > best_score:
                best_score = score
                best_x = x_opt.copy()
                
        except Exception as e:
            pass
            
    if best_x is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        if len(grid) < N:
             # Pad with random?
             # This case shouldn't happen
             best_x = np.random.uniform(-0.4, 0.4, (N, 2))
        else:
             best_x = grid[:N]
             
    return best_x
