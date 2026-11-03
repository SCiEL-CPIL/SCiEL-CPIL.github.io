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

def _relax_circles(circles, w, h, max_iters=500, tol=1e-8):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    circles: (n, 3) array [x, y, r]
    Returns modified circles array.
    This is adapted from the inspiration to ensure we have valid starting points.
    """
    n = circles.shape[0]
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    for _ in range(max_iters):
        moved = False
        
        # 1. Boundary constraints: push circles inside [0, w] x [0, h]
        # Vectorized boundary check
        # x - r < 0 -> x = r
        # x + r > w -> x = w - r
        left_viol = rad - pos[:, 0]
        right_viol = pos[:, 0] + rad - w
        bottom_viol = rad - pos[:, 1]
        top_viol = pos[:, 1] + rad - h
        
        # We need to be careful: if both left and right violate, it's impossible.
        # But for relaxation, we just push.
        # If x < r, set x = r. If x > w-r, set x = w-r.
        # This can oscillate if 2r > w. We assume r < min(w,h)/2.
        
        new_x = pos[:, 0].copy()
        new_y = pos[:, 1].copy()
        
        # Simple clamp logic
        # If x < r, x = r
        # If x > w - r, x = w - r
        # Note: if r > w/2, this is problematic. We bound r in optimizer.
        
        mask_left = new_x < rad
        mask_right = new_x > w - rad
        mask_bottom = new_y < rad
        mask_top = new_y > h - rad
        
        if np.any(mask_left):
            new_x[mask_left] = rad[mask_left]
            moved = True
        if np.any(mask_right):
            new_x[mask_right] = w - rad[mask_right]
            moved = True
        if np.any(mask_bottom):
            new_y[mask_bottom] = rad[mask_bottom]
            moved = True
        if np.any(mask_top):
            new_y[mask_top] = h - rad[mask_top]
            moved = True
            
        pos[:, 0] = new_x
        pos[:, 1] = new_y
        
        # 2. Pairwise collisions: push apart
        # Vectorized pairwise distance calculation
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        r_sum = rad[:, np.newaxis] + rad[np.newaxis, :]
        
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        
        # Get upper triangle indices
        i_idx, j_idx = np.where(mask)
        
        # Distances for pairs
        d_sq_pairs = dist_sq[i_idx, j_idx]
        r_sum_pairs = r_sum[i_idx, j_idx]
        
        # Identify overlapping pairs
        overlap_mask = d_sq_pairs < r_sum_pairs**2
        
        if np.any(overlap_mask):
            # Get indices of overlapping pairs
            oi_idx = i_idx[overlap_mask]
            oj_idx = j_idx[overlap_mask]
            
            # Calculate push vectors
            # diff[i, j] = pos[i] - pos[j]
            # We want to push i away from j, and j away from i.
            # Direction from j to i: (pos[i] - pos[j]) / dist
            # Push i by +dir * overlap/2
            # Push j by -dir * overlap/2
            
            d_pairs = np.sqrt(d_sq_pairs[overlap_mask])
            # Avoid division by zero
            d_pairs_safe = np.where(d_pairs < 1e-9, 1e-9, d_pairs)
            
            # Unit vectors from j to i
            # diff[oi, oj] is pos[oi] - pos[oj]
            vecs = diff[oi_idx, oj_idx]
            vecs_norm = vecs / d_pairs_safe[:, np.newaxis]
            
            overlaps = r_sum_pairs[overlap_mask] - d_pairs
            push_mag = overlaps * 0.5
            
            # Update positions
            # pos[oi] += vecs_norm * push_mag
            # pos[oj] -= vecs_norm * push_mag
            
            # Use np.add.at for unique index updates
            for k in range(len(oi_idx)):
                pos[oi_idx[k]] += vecs_norm[k] * push_mag[k]
                pos[oj_idx[k]] -= vecs_norm[k] * push_mag[k]
            
            moved = True
            
        if not moved:
            break
            
    # Final clamp to ensure boundary validity
    pos[:, 0] = np.clip(pos[:, 0], rad, w - rad)
    pos[:, 1] = np.clip(pos[:, 1], rad, h - rad)
        
    return np.column_stack((pos, rad))

def _objective(x, w, h, penalty_weight=1000.0):
    n = N
    pos = x[:n*2].reshape(n, 2)
    radii = x[n*2:]
    
    # Reuse violation logic but inline for speed in objective
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

def _generate_hex_start(w, h, num_circles, seed=42):
    """
    Generate a hexagonal lattice packing as initial configuration.
    """
    rng = np.random.RandomState(seed)
    r_est = np.sqrt(0.9069 * w * h / (21 * np.pi))
    a = 2 * r_est
    
    points = []
    # Generate points in a grid
    # Hex spacing
    dx = a
    dy = a * np.sqrt(3) / 2
    
    # We want to cover the rectangle
    x_range = np.arange(0, w + a, dx)
    y_range = np.arange(0, h + a, dy)
    
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            # Offset odd columns
            if i % 2 == 1:
                y_offset = y + dy / 2
            else:
                y_offset = y
            if 0 <= x <= w and 0 <= y_offset <= h:
                points.append([x, y_offset])
                
    points = np.array(points)
    
    if len(points) < num_circles:
        # Fallback to random if hex grid is too sparse
        return _random_start(w, h, num_circles, seed)
        
    # Select points closest to center
    cx, cy = w/2, h/2
    dist_to_center = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
    sorted_idx = np.argsort(dist_to_center)
    points = points[sorted_idx[:num_circles]]
    
    # Shuffle to break symmetry for optimizer
    order = rng.permutation(num_circles)
    points = points[order]
    
    radii = np.full(num_circles, r_est * 0.95)
    return points, radii

def _random_start(w, h, num_circles, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
        
    pos = rng.rand(num_circles, 2) * np.array([w, h])
    max_r = min(w, h) / 2.0
    radii = rng.rand(num_circles) * 0.1 * max_r + 0.01 * max_r
    return pos, radii

def _pack_for_aspect(w, h, start_time, deadline):
    best_score = 0
    best_pos = None
    best_radii = None
    
    max_r = min(w, h) / 2.0
    bounds = []
    for i in range(N):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((1e-6, max_r))
        
    initializations = []
    
    # 1. Hexagonal lattice start
    if time.time() < deadline - 20:
        pos_hex, rad_hex = _generate_hex_start(w, h, N, seed=42)
        circles_hex = np.column_stack([pos_hex, rad_hex])
        # Relax to ensure validity
        relaxed_hex = _relax_circles(circles_hex, w, h, max_iters=200)
        x_hex = relaxed_hex.flatten()
        initializations.append(('hex', x_hex))
        
        # Check initial validity
        viol, score, valid = _compute_violations_and_score(relaxed_hex[:, :2], relaxed_hex[:, 2], w, h)
        if valid and score > best_score:
            best_score = score
            best_pos = relaxed_hex[:, :2].copy()
            best_radii = relaxed_hex[:, 2].copy()
            
    # 2. Random restarts with Relaxation
    if time.time() < deadline - 10:
        for seed in [123, 456, 789, 1011, 2023, 3000, 4000, 5000]:
            if time.time() > deadline - 5:
                break
            pos_rand, rad_rand = _random_start(w, h, N, seed=seed)
            circles_rand = np.column_stack([pos_rand, rad_rand])
            # Relax to get a valid start
            relaxed_rand = _relax_circles(circles_rand, w, h, max_iters=100)
            x_rand = relaxed_rand.flatten()
            initializations.append(('rand', x_rand))
            
            # Update best if this relaxed start is better
            viol, score, valid = _compute_violations_and_score(relaxed_rand[:, :2], relaxed_rand[:, 2], w, h)
            if valid and score > best_score:
                best_score = score
                best_pos = relaxed_rand[:, :2].copy()
                best_radii = relaxed_rand[:, 2].copy()

    for idx, (name, x0) in enumerate(initializations):
        if time.time() > deadline - 3:
            break
            
        try:
            # Use L-BFGS-B
            # We start from a valid point, so penalty is 0 initially.
            # We want to maximize radii, so we minimize -sum(radii) + penalty.
            result = minimize(_objective, x0, args=(w, h, 1000.0), method='L-BFGS-B', 
                              bounds=bounds, options={'maxiter': 150, 'ftol': 1e-12, 'maxls': 50})
            
            x_opt = result.x
            pos_opt = x_opt[:N*2].reshape(N, 2)
            rad_opt = x_opt[N*2:]
            
            viol, score, valid = _compute_violations_and_score(pos_opt, rad_opt, w, h)
            
            if not valid:
                # Try to fix with relaxation
                circles_opt = np.column_stack([pos_opt, rad_opt])
                fixed_circles = _relax_circles(circles_opt, w, h, max_iters=100)
                pos_fixed = fixed_circles[:, :2]
                rad_fixed = fixed_circles[:, 2]
                viol_fixed, score_fixed, valid_fixed = _compute_violations_and_score(pos_fixed, rad_fixed, w, h)
                
                if valid_fixed:
                    pos_opt = pos_fixed
                    rad_opt = rad_fixed
                    score = score_fixed
                    valid = True
                else:
                    # If still invalid, discard
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
    # Safety margin: 150s limit, leave 10s buffer
    deadline = start_time + 140
    
    best_global_score = 0
    best_global_result = None
    
    np.random.seed(42)
    
    # Try a range of aspect ratios
    # Perimeter 4 => w + h = 2
    # Focus on ratios that might be better for 21 circles
    # 21 is 3x7, 4x5+1, etc.
    aspect_ratios = np.linspace(0.7, 1.4, 15)
    
    # Ensure square is included
    if not np.any(np.isclose(aspect_ratios, 1.0)):
        aspect_ratios = np.insert(aspect_ratios, len(aspect_ratios)//2, 1.0)
        
    for ar in aspect_ratios:
        if time.time() > deadline - 5:
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
                # Try to fix
                circles = np.column_stack([pos, radii])
                fixed = _relax_circles(circles, w, h, max_iters=200)
                pos_fix = fixed[:, :2]
                rad_fix = fixed[:, 2]
                viol_fix, s_fix, valid_fix = _compute_violations_and_score(pos_fix, rad_fix, w, h)
                if valid_fix:
                    best_global_result = np.column_stack([pos_fix, rad_fix])
                    best_global_score = s_fix
    
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
