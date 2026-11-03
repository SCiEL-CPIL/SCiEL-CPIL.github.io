import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _compute_violations_and_score(pos, radii, w, h):
    """
    Compute constraint violations and the sum of radii.
    Returns (violation_sum, sum_radii, valid_flag)
    """
    n = len(radii)
    d_left = pos[:, 0]
    d_right = w - pos[:, 0]
    d_bottom = pos[:, 1]
    d_top = h - pos[:, 1]
    
    wall_viol = np.sum(np.maximum(0, radii - d_left)**2) + \
                np.sum(np.maximum(0, radii - d_right)**2) + \
                np.sum(np.maximum(0, radii - d_bottom)**2) + \
                np.sum(np.maximum(0, radii - d_top)**2)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    d_sq_upper = dist_sq[mask]
    r_sum_upper = r_sum[mask]
    
    overlap_sq = np.zeros_like(d_sq_upper)
    overlap_mask = d_sq_upper < r_sum_upper**2
    if np.any(overlap_mask):
        d_overlap = np.sqrt(d_sq_upper[overlap_mask])
        overlap_val = r_sum_upper[overlap_mask] - d_overlap
        overlap_sq[overlap_mask] = overlap_val**2
        
    circle_viol = np.sum(overlap_sq)
    
    total_viol = wall_viol + circle_viol
    sum_rad = np.sum(radii)
    valid = total_viol < 1e-6
    return total_viol, sum_rad, valid

def _objective(x, w, h, penalty_weight=100.0):
    n = N
    pos = x[:n*2].reshape(n, 2)
    radii = x[n*2:]
    
    d_left = pos[:, 0]
    d_right = w - pos[:, 0]
    d_bottom = pos[:, 1]
    d_top = h - pos[:, 1]
    
    wall_viol = np.sum(np.maximum(0, radii - d_left)**2) + \
                np.sum(np.maximum(0, radii - d_right)**2) + \
                np.sum(np.maximum(0, radii - d_bottom)**2) + \
                np.sum(np.maximum(0, radii - d_top)**2)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    d_sq_upper = dist_sq[mask]
    r_sum_upper = r_sum[mask]
    
    overlap_sq = np.zeros_like(d_sq_upper)
    overlap_mask = d_sq_upper < r_sum_upper**2
    if np.any(overlap_mask):
        d_overlap = np.sqrt(d_sq_upper[overlap_mask])
        overlap_val = r_sum_upper[overlap_mask] - d_overlap
        overlap_sq[overlap_mask] = overlap_val**2
        
    circle_viol = np.sum(overlap_sq)
    total_viol = wall_viol + circle_viol
    sum_rad = np.sum(radii)
    
    return penalty_weight * total_viol - sum_rad

def _greedy_place(w, h, num_circles, grid_res=50):
    """
    Place circles sequentially with better initialization.
    """
    pos = np.zeros((num_circles, 2))
    radii = np.zeros(num_circles)
    
    # Finer grid for better initial placement
    xs = np.linspace(0.02*w, 0.98*w, grid_res)
    ys = np.linspace(0.02*h, 0.98*h, grid_res)
    
    for k in range(num_circles):
        best_r = 0
        best_x = w / 2.0
        best_y = h / 2.0
        
        XX, YY = np.meshgrid(xs, ys)
        
        r_wall = np.minimum(np.minimum(XX, w - XX), np.minimum(YY, h - YY))
        
        r_circle = np.full_like(r_wall, np.inf)
        for i in range(k):
            dx = XX - pos[i, 0]
            dy = YY - pos[i, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            r_max_i = dist - radii[i]
            r_circle = np.minimum(r_circle, r_max_i)
            
        r_allowed = np.minimum(r_wall, r_circle)
        
        max_idx = np.unravel_index(np.argmax(r_allowed), r_allowed.shape)
        best_r = r_allowed[max_idx]
        best_y = ys[max_idx[0]]
        best_x = xs[max_idx[1]]
        
        pos[k, 0] = best_x
        pos[k, 1] = best_y
        radii[k] = best_r
        
    return pos, radii

def _random_start(w, h, seed=None):
    """Generate a random initial configuration with reasonable radii."""
    if seed is not None:
        np.random.seed(seed)
    
    pos = np.random.rand(N, 2) * np.array([w, h])
    # Start with moderately large radii to give optimizer a better starting point
    base_r = 0.08 * min(w, h)
    rad = np.random.uniform(0.5, 1.5, N) * base_r
    return pos, rad

def _polish(pos, radii, w, h, deadline):
    """
    High-precision polish pass with tighter tolerances and more iterations.
    """
    n = N
    x0 = np.concatenate([pos.flatten(), radii])
    
    max_r = min(w, h) / 2.0
    bounds = []
    for i in range(n):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0, max_r))
    
    # Use higher penalty and tighter tolerance for polish
    best_score = np.sum(radii)
    best_x = x0.copy()
    
    # Try with increasing precision
    for penalty in [200.0, 500.0]:
        if time.time() > deadline - 1:
            break
        try:
            result = minimize(_objective, best_x, args=(w, h, penalty), method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 500, 'ftol': 1e-14, 'maxls': 30})
            
            x_opt = result.x
            pos_opt = x_opt[:n*2].reshape(n, 2)
            rad_opt = x_opt[n*2:]
            
            viol, score, valid = _compute_violations_and_score(pos_opt, rad_opt, w, h)
            if valid and score > best_score:
                best_score = score
                best_x = x_opt.copy()
        except Exception:
            continue
    
    # Final very tight polish
    if time.time() < deadline - 1:
        try:
            result = minimize(_objective, best_x, args=(w, h, 1000.0), method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-16, 'maxls': 50})
            x_opt = result.x
            pos_opt = x_opt[:n*2].reshape(n, 2)
            rad_opt = x_opt[n*2:]
            
            viol, score, valid = _compute_violations_and_score(pos_opt, rad_opt, w, h)
            if valid and score > best_score:
                best_score = score
                best_x = x_opt.copy()
        except Exception:
            pass
    
    final_pos = best_x[:n*2].reshape(n, 2)
    final_rad = best_x[n*2:]
    return final_pos, final_rad, best_score

def _pack_for_aspect(w, h, start_time, deadline):
    """
    Pack circles in w x h rectangle with improved strategy.
    """
    best_score = 0
    best_pos = None
    best_radii = None
    
    max_r = min(w, h) / 2.0
    bounds = []
    for i in range(N):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0, max_r))
        
    # 1. Greedy Initialization (finer grid)
    pos_grid, rad_grid = _greedy_place(w, h, N, grid_res=50)
    x0 = np.concatenate([pos_grid.flatten(), rad_grid])
    
    viol, score, valid = _compute_violations_and_score(pos_grid, rad_grid, w, h)
    if valid and score > best_score:
        best_score = score
        best_pos = pos_grid.copy()
        best_radii = rad_grid.copy()
        
    # 2. Multiple initializations: greedy + several random restarts
    initializations = [x0]
    
    # Add random restarts with different seeds
    if time.time() < deadline - 10:
        for seed in [42, 123, 456]:
            pos_rand, rad_rand = _random_start(w, h, seed=seed)
            x_rand = np.concatenate([pos_rand.flatten(), rad_rand])
            initializations.append(x_rand)

    for idx, x_init in enumerate(initializations):
        if time.time() > deadline - 5:
            break
            
        try:
            # Adaptive penalty: start moderate, increase if violations persist
            result = minimize(_objective, x_init, args=(w, h, 100.0), method='L-BFGS-B', 
                              bounds=bounds, options={'maxiter': 300, 'ftol': 1e-12, 'maxls': 20})
            
            x_opt = result.x
            pos_opt = x_opt[:N*2].reshape(N, 2)
            rad_opt = x_opt[N*2:]
            
            viol, score, valid = _compute_violations_and_score(pos_opt, rad_opt, w, h)
            
            if not valid and viol > 1e-4:
                # Try to fix by scaling down radii
                scale = 0.95
                while viol > 1e-6 and scale > 0.1:
                    rad_new = rad_opt * scale
                    viol, s, valid = _compute_violations_and_score(pos_opt, rad_new, w, h)
                    if valid:
                        rad_opt = rad_new
                        score = s
                        break
                    scale *= 0.9
                if not valid:
                    continue
            else:
                score = np.sum(rad_opt)
                
            if score > best_score:
                best_score = score
                best_pos = pos_opt.copy()
                best_radii = rad_opt.copy()
                
        except Exception:
            continue
            
    # 3. Polish the best result
    if best_pos is not None and time.time() < deadline - 3:
        polished_pos, polished_rad, polished_score = _polish(best_pos, best_radii, w, h, deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_pos = polished_pos
            best_radii = polished_rad
            
    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    # 150s limit. Use 145s for safety.
    deadline = start_time + 145
    
    best_global_score = 0
    best_global_result = None
    
    np.random.seed(42)
    
    # Focus on promising aspect ratios
    # Square and slightly elongated are typically best
    aspect_ratios = [1.0, 0.9, 1.1, 0.85, 1.15, 0.8, 1.2]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        pos, radii, score = _pack_for_aspect(w, h, start_time, deadline)
        
        if pos is not None and score > best_global_score:
            best_global_score = score
            viol, s, valid = _compute_violations_and_score(pos, radii, w, h)
            if valid:
                best_global_result = np.column_stack([pos, radii])
            elif viol < 1e-3:
                scale = 0.99
                while viol > 1e-6 and scale > 0.1:
                    radii_new = radii * scale
                    viol, s, valid = _compute_violations_and_score(pos, radii_new, w, h)
                    if valid:
                        radii = radii_new
                        break
                    scale *= 0.95
                if valid:
                    best_global_result = np.column_stack([pos, radii])
    
    # Fallback
    if best_global_result is None:
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        pos = np.zeros((N, 2))
        rad = np.zeros(N)
        r_val = 0.5 / side
        for i in range(N):
            col = i % side
            row = i // side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
            rad[i] = r_val * 0.9
        best_global_result = np.column_stack([pos, rad])
        
    return best_global_result
