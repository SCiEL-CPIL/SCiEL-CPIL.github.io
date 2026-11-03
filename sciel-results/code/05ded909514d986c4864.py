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
    # Vectorized wall violations
    d_left = pos[:, 0]
    d_right = w - pos[:, 0]
    d_bottom = pos[:, 1]
    d_top = h - pos[:, 1]
    
    wall_viol = np.sum(np.maximum(0, radii - d_left)**2) + \
                np.sum(np.maximum(0, radii - d_right)**2) + \
                np.sum(np.maximum(0, radii - d_bottom)**2) + \
                np.sum(np.maximum(0, radii - d_top)**2)
    
    # Circle-Circle violations
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
    valid = total_viol < 1e-8
    return total_viol, sum_rad, valid

def _objective(x, w, h, penalty_weight=1000.0):
    """
    Objective function for optimization.
    x = [x1, y1, r1, x2, y2, r2, ...]
    We minimize: penalty_weight * (Sum of Squared Violations) - (Sum of Radii)
    """
    n = N
    pos = x[:n*2].reshape(n, 2)
    radii = x[n*2:]
    
    # Compute violations
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
    Place circles sequentially to maximize the minimum radius at each step.
    Returns pos (N, 2) and radii (N,)
    """
    pos = np.zeros((num_circles, 2))
    radii = np.zeros(num_circles)
    
    # Grid of potential centers
    xs = np.linspace(0.02*w, 0.98*w, grid_res)
    ys = np.linspace(0.02*h, 0.98*h, grid_res)
    
    for k in range(num_circles):
        best_r = 0
        best_x = w / 2.0
        best_y = h / 2.0
        
        # Create meshgrid
        XX, YY = np.meshgrid(xs, ys)
        
        # Wall radius
        r_wall = np.minimum(np.minimum(XX, w - XX), np.minimum(YY, h - YY))
        
        # Circle radius constraint
        r_circle = np.full_like(r_wall, np.inf)
        for i in range(k):
            dx = XX - pos[i, 0]
            dy = YY - pos[i, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            r_max_i = dist - radii[i]
            r_circle = np.minimum(r_circle, r_max_i)
            
        # Max allowed radius at each grid point
        r_allowed = np.minimum(r_wall, r_circle)
        
        # Find max
        max_idx = np.unravel_index(np.argmax(r_allowed), r_allowed.shape)
        best_r = r_allowed[max_idx]
        best_y = ys[max_idx[0]]
        best_x = xs[max_idx[1]]
        
        pos[k, 0] = best_x
        pos[k, 1] = best_y
        radii[k] = best_r
        
    return pos, radii

def _random_start(w, h, num_circles, seed=None):
    """
    Generate a random initial configuration.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    pos = rng.rand(num_circles, 2) * np.array([w, h])
    # Start with small radii to avoid huge overlaps
    max_r = min(w, h) / 2.0
    radii = rng.rand(num_circles) * 0.1 * max_r
    
    return pos, radii

def _pack_for_aspect(w, h, start_time, deadline):
    """
    Pack circles in w x h rectangle.
    Returns best (pos, radii) and score found for this aspect.
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
        
    # 1. Greedy Initialization
    pos_grid, rad_grid = _greedy_place(w, h, N, grid_res=50)
    x0 = np.concatenate([pos_grid.flatten(), rad_grid])
    
    # Check initial validity and score
    viol, score, valid = _compute_violations_and_score(pos_grid, rad_grid, w, h)
    if valid and score > best_score:
        best_score = score
        best_pos = pos_grid.copy()
        best_radii = rad_grid.copy()
        
    # 2. Optimization
    # We will run L-BFGS-B from the greedy start and some random restarts.
    
    initializations = [x0]
    
    # Add a few random restarts if we have time
    if time.time() < deadline - 10:
        for seed in [42, 123, 456]:
            if time.time() > deadline - 5:
                break
            pos_rand, rad_rand = _random_start(w, h, N, seed=seed)
            x_rand = np.concatenate([pos_rand.flatten(), rad_rand])
            initializations.append(x_rand)

    for idx, x0 in enumerate(initializations):
        if time.time() > deadline - 3:
            break
            
        try:
            # Use a high penalty weight to ensure feasibility
            # Then we can refine with lower penalty if needed, but usually high is better for packing
            result = minimize(_objective, x0, args=(w, h, 1000.0), method='L-BFGS-B', 
                              bounds=bounds, options={'maxiter': 500, 'ftol': 1e-14, 'maxls': 50})
            
            x_opt = result.x
            pos_opt = x_opt[:N*2].reshape(N, 2)
            rad_opt = x_opt[N*2:]
            
            # Verify and clean up
            viol, score, valid = _compute_violations_and_score(pos_opt, rad_opt, w, h)
            
            if not valid:
                # Try to fix by scaling down radii
                if viol > 1e-4:
                    scale = 0.95
                    while viol > 1e-8 and scale > 0.1:
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
            
    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    # 150s limit. Use 130s for safety.
    deadline = start_time + 130
    
    best_global_score = 0
    best_global_result = None
    
    # Aspect ratios to try
    # Perimeter 4 => 2(w+h)=4 => w+h=2 => h = 2-w
    # w in [0.5, 1.5]
    # Square is w=1, h=1.
    # We will try a range of aspect ratios.
    # The optimal aspect ratio for packing is often slightly elongated.
    
    np.random.seed(42)
    
    # Generate aspect ratios w/h
    # w = 2*ar/(1+ar), h = 2/(1+ar)
    # ar = w/h
    # Let's try ar from 0.7 to 1.4
    aspect_ratios = np.linspace(0.7, 1.4, 15)
    
    # Ensure we check the square (ar=1.0)
    if not np.any(np.isclose(aspect_ratios, 1.0)):
        aspect_ratios = np.insert(aspect_ratios, len(aspect_ratios)//2, 1.0)
        
    for ar in aspect_ratios:
        if time.time() > deadline - 5: # Leave 5s for cleanup
            break
            
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        pos, radii, score = _pack_for_aspect(w, h, start_time, deadline)
        
        if pos is not None and score > best_global_score:
            best_global_score = score
            # Ensure strict validity
            viol, s, valid = _compute_violations_and_score(pos, radii, w, h)
            if valid:
                best_global_result = np.column_stack([pos, radii])
            elif viol < 1e-3:
                # Fix by shrinking
                scale = 0.99
                while viol > 1e-8 and scale > 0.1:
                    radii_new = radii * scale
                    viol, s, valid = _compute_violations_and_score(pos, radii_new, w, h)
                    if valid:
                        radii = radii_new
                        break
                    scale *= 0.95
                if valid:
                    best_global_result = np.column_stack([pos, radii])
    
    # Fallback if no valid result found
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
