import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_radius(x):
    """Compute the maximum radius for the given configuration."""
    # Pair distances
    dx = x[:, 0:1] - x[:, 0:1].T
    dy = x[:, 1:2] - x[:, 1:2].T
    dists_sq = dx**2 + dy**2
    np.fill_diagonal(dists_sq, 1.0)
    min_dist = np.sqrt(np.min(dists_sq))
    
    # Wall distances
    wx = 0.5 - np.abs(x[:, 0])
    wy = 0.5 - np.abs(x[:, 1])
    min_wall = np.min(np.minimum(wx, wy))
    
    return min(min_dist / 2.0, min_wall)

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
    max_val = np.max(vals)
    exp_vals = np.exp(-beta * (vals - max_val))
    sum_exp = np.sum(exp_vals)
    softmin_val = max_val - (1.0 / beta) * np.log(sum_exp)
    
    # We want to maximize softmin_val.
    # So we minimize -softmin_val.
    obj_val = -softmin_val
    
    # Gradient of softmin wrt vals
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
        g_soft[i] += p_k * (diff / d)
        g_soft[j] -= p_k * (diff / d)
        
    # Walls
    for i in range(N):
        p_k = wall_p[i]
        if p_k < 1e-15:
            continue
        wx_i = wx[i]
        wy_i = wy[i]
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

def _init_hex(r_guess=0.07):
    """Initialize points in a hexagonal pattern."""
    dx = 2 * r_guess
    dy = np.sqrt(3) * r_guess
    
    points = []
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
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]
    else:
        # Pad if necessary
        current_len = len(points)
        if current_len < N:
            pad = np.random.uniform(-0.45, 0.45, (N - current_len, 2))
            points = np.vstack([points, pad])
        return points

def _simulated_annealing(x0, start_time, deadline, steps=5000, t_start=0.05, t_end=0.0001):
    """Run simulated annealing to improve the packing."""
    x = x0.copy()
    best_x = x.copy()
    best_r = _compute_radius(x)
    
    current_r = best_r
    step = 0
    while step < steps and time.time() < deadline:
        # Update temperature
        if step < steps:
            t = t_start * (t_end / t_start) ** (step / steps)
        else:
            t = t_end
            
        # Pick a random circle
        i = np.random.randint(N)
        
        # Generate a random move
        move_mag = 0.01 * np.sqrt(t / t_start) + 0.001
        dx = np.random.normal(0, move_mag)
        dy = np.random.normal(0, move_mag)
        
        x_new = x.copy()
        x_new[i, 0] += dx
        x_new[i, 1] += dy
        
        # Check bounds
        if x_new[i, 0] < -0.5 or x_new[i, 0] > 0.5 or x_new[i, 1] < -0.5 or x_new[i, 1] > 0.5:
            step += 1
            continue
            
        new_r = _compute_radius(x_new)
        
        # Accept if better or with probability exp(-(new_r - current_r)/t)
        if new_r > current_r:
            x = x_new
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_x = x.copy()
        else:
            delta = new_r - current_r
            if t > 1e-10:
                prob = np.exp(delta / t)
            else:
                prob = 0.0
                
            if np.random.rand() < prob:
                x = x_new
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_x = x.copy()
                    
        step += 1
        
    return best_x, best_r

def _lbfgs_polish(x0, start_time, deadline):
    """Polish the configuration using L-BFGS-B."""
    beta = 100.0
    
    try:
        res = minimize(
            _softmin_obj_and_grad,
            x0,
            args=(beta,),
            method='L-BFGS-B',
            jac=True,
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        x_opt = res.x.reshape(N, 2)
        
        if not np.all(np.isfinite(x_opt)):
            return x0, _compute_radius(x0)
            
        x_opt = np.clip(x_opt, -0.5, 0.5)
        
        score = _compute_radius(x_opt)
        return x_opt, score
    except Exception:
        return x0, _compute_radius(x0)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_x = None
    best_score = -np.inf
    
    num_restarts = 10
    
    for k in range(num_restarts):
        if time.time() > deadline:
            break
            
        if k == 0:
            x0 = _init_hex(0.07)
        elif k % 2 == 1:
            x0 = _init_hex(0.06)
        else:
            x0 = np.random.uniform(-0.45, 0.45, (N, 2))
            
        if not np.all(np.isfinite(x0)):
            continue
            
        sa_steps = 2000
        x_sa, r_sa = _simulated_annealing(x0, start_time, deadline, steps=sa_steps)
        
        x_polished, r_polished = _lbfgs_polish(x_sa, start_time, deadline)
        
        if r_polished > best_score:
            best_score = r_polished
            best_x = x_polished.copy()
            
    if best_x is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        if len(grid) < N:
            best_x = np.random.uniform(-0.4, 0.4, (N, 2))
        else:
            best_x = grid[:N]
             
    return best_x
