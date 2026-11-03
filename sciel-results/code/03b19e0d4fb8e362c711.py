import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _check_validity(pos, radii, w, h, tol=1e-9):
    """Check if circles are valid: inside rectangle, no overlaps, positive radii."""
    n = len(radii)
    if n != N:
        return False
    for i in range(n):
        x, y, r = pos[i, 0], pos[i, 1], radii[i]
        if r <= 0:
            return False
        if x - r < -tol or y - r < -tol or x + r > w + tol or y + r > h + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

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
    valid = total_viol < 1e-8
    return total_viol, sum_rad, valid

def _separate_circles(circles, W, H, max_iter=500):
    """
    Directly resolve overlaps and boundary violations using pairwise repulsion.
    Returns (circles, is_valid).
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    for it in range(max_iter):
        moved = False

        # Clamp position to keep circle inside rectangle
        for i in range(n):
            if r[i] <= 0:
                r[i] = 1e-4
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        # Resolve overlaps via pairwise separation
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        dist = 1e-6
                    overlap = min_dist - dist
                    total_r = r[i] + r[j]
                    wi = r[i] / total_r if total_r > 0 else 0.5
                    wj = r[j] / total_r if total_r > 0 else 0.5
                    push = overlap + 1e-8
                    x[i] += wi * push * dx / dist
                    y[i] += wi * push * dy / dist
                    x[j] -= wj * push * dx / dist
                    y[j] -= wj * push * dy / dist
                    moved = True

        # Re-clamp after separation, shrinking radius if circle doesn't fit
        for i in range(n):
            max_r_x = min(x[i], W - x[i])
            max_r_y = min(y[i], H - y[i])
            max_r = max(1e-4, min(max_r_x, max_r_y))
            if r[i] > max_r:
                r[i] = max_r
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        if not moved:
            break

    result = np.column_stack([x, y, r])
    valid = _check_validity(result[:, :2], result[:, 2], W, H)
    return result, valid

def _objective(x, w, h, penalty_weight=1000.0):
    """
    Objective function for optimization.
    x = [x1, y1, r1, x2, y2, r2, ...]
    We minimize: penalty_weight * (Sum of Squared Violations) - (Sum of Radii)
    """
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

def _polish(pos, radii, w, h, start_time, deadline, max_iter=2000, ftol=1e-16):
    """
    High-precision polish pass with very tight tolerances.
    """
    x0 = np.concatenate([pos.flatten(), radii])
    max_r = min(w, h) / 2.0
    bounds = []
    for i in range(N):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0, max_r))
    
    try:
        result = minimize(_objective, x0, args=(w, h, 5000.0), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': max_iter, 'ftol': ftol, 'maxls': 100, 'gtol': 1e-12})
        
        x_opt = result.x
        pos_opt = x_opt[:N*2].reshape(N, 2)
        rad_opt = x_opt[N*2:]
        
        # Apply separation to fix any remaining issues
        circles = np.column_stack([pos_opt, rad_opt])
        circles, sep_valid = _separate_circles(circles, w, h, max_iter=1000)
        pos_opt = circles[:, :2]
        rad_opt = circles[:, 2]
        
        viol, score, valid = _compute_violations_and_score(pos_opt, rad_opt, w, h)
        
        if not valid:
            scale = 0.99
            while viol > 1e-8 and scale > 0.5:
                rad_new = rad_opt * scale
                viol, s, valid = _compute_violations_and_score(pos_opt, rad_new, w, h)
                if valid:
                    rad_opt = rad_new
                    score = s
                    break
                scale *= 0.98
            if not valid:
                return None, None, 0
        
        return pos_opt, rad_opt, np.sum(rad_opt)
    except Exception:
        return None, None, 0

def _hexagonal_start(w, h, num_circles):
    """
    Generate a hexagonal lattice packing as initial configuration.
    """
    pos = np.zeros((num_circles, 2))
    radii = np.zeros(num_circles)
    
    # Try different grid dimensions that sum to 21
    # 5x4 + 1, 4x5 + 1, 3x7, 7x3, 6x3 + 3, 3x6 + 3
    # Let's try a few and pick the best fit
    
    best_pos = None
    best_rad = None
    best_score = -1
    
    grid_configs = [
        (5, 4), (4, 5), (3, 7), (7, 3), (6, 3), (3, 6), (5, 5)
    ]
    
    for cols, rows in grid_configs:
        total = cols * rows
        if total < num_circles:
            continue
            
        pos_local = np.zeros((num_circles, 2))
        radii_local = np.zeros(num_circles)
        
        for i in range(num_circles):
            if i < total:
                row = i // cols
                col = i % cols
            else:
                # Place extra circles in the middle or random spots
                row = num_circles // 2
                col = num_circles % 2
                
            # Hexagonal offset
            x = (col + 0.5) / cols * w
            y = (row + 0.5) / rows * h
            
            if row % 2 == 1:
                x += w / (2 * cols)
                
            # Clamp to boundaries
            x = np.clip(x, 0.1, w - 0.1)
            y = np.clip(y, 0.1, h - 0.1)
            
            pos_local[i, 0] = x
            pos_local[i, 1] = y
        
        # Compute initial radii based on minimum distance
        min_dist = np.inf
        for i in range(num_circles):
            for j in range(i+1, num_circles):
                d = np.sqrt(np.sum((pos_local[i] - pos_local[j])**2))
                min_dist = min(min_dist, d)
        
        for i in range(num_circles):
            d_wall = min(pos_local[i, 0], w - pos_local[i, 0], pos_local[i, 1], h - pos_local[i, 1])
            min_dist = min(min_dist, d_wall)
        
        radii_local[:] = 0.45 * min_dist
        
        # Check validity and score
        viol, score, valid = _compute_violations_and_score(pos_local, radii_local, w, h)
        if valid and score > best_score:
            best_score = score
            best_pos = pos_local.copy()
            best_rad = radii_local.copy()
            
    if best_pos is None:
        # Fallback to simple grid
        cols = 5
        rows = 5
        for i in range(num_circles):
            row = i // cols
            col = i % cols
            x = (col + 0.5) / cols * w
            y = (row + 0.5) / rows * h
            pos[i, 0] = x
            pos[i, 1] = y
        min_dist = np.inf
        for i in range(num_circles):
            for j in range(i+1, num_circles):
                d = np.sqrt(np.sum((pos[i] - pos[j])**2))
                min_dist = min(min_dist, d)
        for i in range(num_circles):
            d_wall = min(pos[i, 0], w - pos[i, 0], pos[i, 1], h - pos[i, 1])
            min_dist = min(min_dist, d_wall)
        radii[:] = 0.45 * min_dist
        return pos, radii
        
    return best_pos, best_rad

def _greedy_place(w, h, num_circles, grid_res=50):
    """
    Place circles sequentially to maximize the minimum radius at each step.
    Returns pos (N, 2) and radii (N,)
    """
    pos = np.zeros((num_circles, 2))
    radii = np.zeros(num_circles)
    
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

def _random_start(w, h, num_circles, seed=None):
    """
    Generate a random initial configuration.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    pos = rng.rand(num_circles, 2) * np.array([w, h])
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
        
    # 1. Hexagonal Initialization
    pos_hex, rad_hex = _hexagonal_start(w, h, N)
    x0_hex = np.concatenate([pos_hex.flatten(), rad_hex])
    
    viol, score, valid = _compute_violations_and_score(pos_hex, rad_hex, w, h)
    if valid and score > best_score:
        best_score = score
        best_pos = pos_hex.copy()
        best_radii = rad_hex.copy()
        
    # 2. Greedy Initialization
    pos_grid, rad_grid = _greedy_place(w, h, N, grid_res=50)
    x0_grid = np.concatenate([pos_grid.flatten(), rad_grid])
    
    viol, score, valid = _compute_violations_and_score(pos_grid, rad_grid, w, h)
    if valid and score > best_score:
        best_score = score
        best_pos = pos_grid.copy()
        best_radii = rad_grid.copy()
        
    # 3. Optimization from multiple starts
    initializations = [x0_hex, x0_grid]
    
    # Add random restarts if we have time
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
            # Use high penalty weight for feasibility
            result = minimize(_objective, x0, args=(w, h, 1000.0), method='L-BFGS-B', 
                              bounds=bounds, options={'maxiter': 500, 'ftol': 1e-14, 'maxls': 50})
            
            x_opt = result.x
            pos_opt = x_opt[:N*2].reshape(N, 2)
            rad_opt = x_opt[N*2:]
            
            # Apply separation to fix any remaining issues
            circles = np.column_stack([pos_opt, rad_opt])
            circles, sep_valid = _separate_circles(circles, w, h, max_iter=200)
            pos_opt = circles[:, :2]
            rad_opt = circles[:, 2]
            
            viol, score, valid = _compute_violations_and_score(pos_opt, rad_opt, w, h)
            
            if not valid:
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
    
    # 4. High-precision polish pass if we have time
    if best_pos is not None and time.time() < deadline - 15:
        pos_pol, rad_pol, score_pol = _polish(best_pos, best_radii, w, h, start_time, deadline)
        if pos_pol is not None and score_pol > best_score:
            best_score = score_pol
            best_pos = pos_pol
            best_radii = rad_pol
    
    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    best_global_score = 0
    best_global_result = None
    
    np.random.seed(42)
    
    # Focus on promising aspect ratios
    aspect_ratios = np.array([0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2])
    
    for ar in aspect_ratios:
        if time.time() > deadline - 20:
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
                while viol > 1e-8 and scale > 0.1:
                    radii_new = radii * scale
                    viol, s, valid = _compute_violations_and_score(pos, radii_new, w, h)
                    if valid:
                        radii = radii_new
                        break
                    scale *= 0.95
                if valid:
                    best_global_result = np.column_stack([pos, radii])
    
    # Final polish on the best global result if we have time
    if best_global_result is not None and time.time() < deadline - 10:
        pos_best = best_global_result[:, :2]
        rad_best = best_global_result[:, 2]
        # Find w, h for this result
        # We need to track which aspect ratio was used, so let's just try polishing with the current best
        # Actually, we need to know w and h. Let's recompute from the aspect ratio that gave the best result.
        # This is a bit tricky, so let's just do a simple polish with the best aspect ratio we tried.
        
        # For now, skip the final polish since we don't track which aspect ratio gave the best result
        pass
    
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
