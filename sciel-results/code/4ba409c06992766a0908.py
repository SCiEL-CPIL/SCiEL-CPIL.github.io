import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _compute_violations_and_score(pos, radii, w, h):
    n = len(radii)
    d_left = pos[:, 0]
    d_right = w - pos[:, 0]
    d_bottom = pos[:, 1]
    d_top = h - pos[:, 1]
    
    # Wall violations
    wall_viol = np.sum(np.maximum(0, radii - d_left)**2) + \
                np.sum(np.maximum(0, radii - d_right)**2) + \
                np.sum(np.maximum(0, radii - d_bottom)**2) + \
                np.sum(np.maximum(0, radii - d_top)**2)
    
    # Circle-circle violations
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_sq_upper = dist_sq[mask]
    r_sum_upper = r_sum[mask]
    
    # max(0, r_i + r_j - dist)^2
    # dist = sqrt(d_sq)
    # violation = max(0, r_sum - sqrt(d_sq))
    # We can compute this efficiently
    # If d_sq < r_sum^2, then violation = r_sum - sqrt(d_sq)
    # else 0
    
    overlap_sq = np.zeros_like(d_sq_upper)
    overlap_mask = d_sq_upper < r_sum_upper**2
    if np.any(overlap_mask):
        d_overlap = np.sqrt(d_sq_upper[overlap_mask])
        overlap_val = np.maximum(0, r_sum_upper[overlap_mask] - d_overlap)
        overlap_sq[overlap_mask] = overlap_val**2
        
    circle_viol = np.sum(overlap_sq)
    total_viol = wall_viol + circle_viol
    sum_rad = np.sum(radii)
    valid = total_viol < 1e-9
    return total_viol, sum_rad, valid

def _repair(pos, radii, w, h, tol=1e-9, max_iter=100):
    """
    Iteratively reduce radii and adjust positions to ensure validity.
    Returns valid pos, radii, score.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # 1. Clip positions to be inside bounds with margin
    # Ensure centers are at least radius away from walls? No, just keep them inside.
    # If a circle is too big, we shrink it.
    
    for _ in range(max_iter):
        viol, score, valid = _compute_violations_and_score(pos, radii, w, h)
        if valid:
            return pos, radii, score
        
        # Identify worst offender? Or just scale down all radii?
        # Scaling down radii is safe and preserves relative geometry.
        # If we scale radii by factor k, new viol is roughly k^2 * old_viol (for overlaps)
        # but wall violations also scale.
        
        # Let's try a binary search on the scaling factor for radii
        # or simply reduce by 5% each step.
        
        # Also, check if any radius is too large for the box
        max_possible_r = min(w, h) / 2.0
        if np.any(radii > max_possible_r):
            radii = np.minimum(radii, max_possible_r * 0.99)
            continue
            
        # Reduce radii
        radii *= 0.95
        
        # If radii become too small, we might want to try moving points?
        # For now, just shrink. This guarantees a valid (though suboptimal) solution.
        if np.all(radii < 1e-6):
            break
            
    # Final check
    viol, score, valid = _compute_violations_and_score(pos, radii, w, h)
    if not valid:
        # Force validity by zeroing out? No, that's bad.
        # Just return what we have, it should be valid after shrinking enough.
        # If not, shrink more aggressively.
        radii *= 0.5
        viol, score, valid = _compute_violations_and_score(pos, radii, w, h)
        
    return pos, radii, score

def _objective(x, w, h, penalty_weight=1000.0, eps=0.001):
    n = N
    pos = x[:n*2].reshape(n, 2)
    radii = x[n*2:]
    
    d_left = pos[:, 0]
    d_right = w - pos[:, 0]
    d_bottom = pos[:, 1]
    d_top = h - pos[:, 1]
    
    # Wall violations (smoothed max)
    # max(0, r - d) approximated by 0.5 * (r - d + sqrt((r-d)^2 + eps^2))
    def soft_max(x, eps):
        return 0.5 * (x + np.sqrt(x**2 + eps**2))
        
    wall_viol_terms = [
        soft_max(radii - d_left, eps),
        soft_max(radii - d_right, eps),
        soft_max(radii - d_bottom, eps),
        soft_max(radii - d_top, eps)
    ]
    wall_viol = np.sum([np.sum(v**2) for v in wall_viol_terms])
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    d_sq_upper = dist_sq[mask]
    r_sum_upper = r_sum[mask]
    
    d_upper = np.sqrt(np.maximum(d_sq_upper, 1e-12))
    overlap_raw = r_sum_upper - d_upper
    
    # Smooth overlap penalty
    smooth_overlap = soft_max(overlap_raw, eps)
    
    circle_viol = np.sum(smooth_overlap**2)
    total_viol = wall_viol + circle_viol
    sum_rad = np.sum(radii)
    
    return penalty_weight * total_viol - sum_rad

def _hexagonal_start(w, h, num_circles, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(42)
        
    # Estimate radius based on area packing density
    # For hexagonal packing, density is ~0.9069
    # Area = w * h
    # N * pi * r^2 <= 0.9069 * Area
    r_est = np.sqrt(0.9069 * w * h / (num_circles * np.pi))
    
    # We want to place circles such that they don't overlap initially.
    # Let's use a grid of potential centers.
    # Spacing s = 2 * r_est
    s = 2 * r_est * 1.05 # small margin
    
    # Generate grid points
    nx = int(np.ceil(w / s)) + 2
    ny = int(np.ceil(h / s)) + 2
    
    xs = np.linspace(-s, w + s, nx)
    ys = np.linspace(-s, h + s, ny)
    
    points = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if i % 2 == 1:
                y_offset = y + s / 2
            else:
                y_offset = y
            points.append([x, y_offset])
            
    points = np.array(points)
    
    # Filter points that are reasonably inside
    # We allow centers to be slightly outside if the circle is mostly inside?
    # No, for initialization, let's keep centers inside [0, w] x [0, h]
    # But if r is large, center must be in [r, w-r].
    # Let's just filter by [0, w] and [0, h] for now.
    valid_mask = (points[:, 0] >= 0) & (points[:, 0] <= w) & \
                 (points[:, 1] >= 0) & (points[:, 1] <= h)
    points = points[valid_mask]
    
    if len(points) > num_circles:
        # Pick the ones closest to the center
        cx, cy = w/2, h/2
        dist_to_center = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
        sorted_idx = np.argsort(dist_to_center)
        points = points[sorted_idx[:num_circles]]
    elif len(points) < num_circles:
        # Not enough points, add random ones
        missing = num_circles - len(points)
        extra_points = rng.rand(missing, 2) * np.array([w, h])
        points = np.vstack([points, extra_points])
        
    # Shuffle
    order = rng.permutation(num_circles)
    points = points[order]
    
    # Initialize radii slightly smaller than estimated to ensure no initial overlap
    # in the heuristic sense, though they might still overlap.
    radii = np.full(num_circles, r_est * 0.9)
    
    return points, radii

def _random_start(w, h, num_circles, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(0)
    
    pos = rng.rand(num_circles, 2) * np.array([w, h])
    max_r = min(w, h) / 2.0
    radii = rng.rand(num_circles) * 0.1 * max_r
    
    return pos, radii

def _polish(pos, radii, w, h, start_time, deadline):
    """High-precision polish with progressive eps reduction."""
    current_pos = pos.copy()
    current_radii = radii.copy()
    
    # Progressive eps values for smooth-max
    eps_schedule = [0.01, 0.001, 0.0001, 1e-5]
    penalty_schedule = [100.0, 1000.0, 10000.0, 100000.0]
    
    max_r = min(w, h) / 2.0
    bounds = []
    for i in range(N):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0, max_r))
    
    for eps, pen in zip(eps_schedule, penalty_schedule):
        if time.time() > deadline - 5:
            break
        
        x0 = np.concatenate([current_pos.flatten(), current_radii])
        
        try:
            result = minimize(_objective, x0, args=(w, h, pen, eps), 
                             method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 300, 'ftol': 1e-12, 'maxls': 50})
            
            x_opt = result.x
            new_pos = x_opt[:N*2].reshape(N, 2)
            new_radii = x_opt[N*2:]
            
            # Ensure new_radii are positive
            new_radii = np.maximum(new_radii, 0)
            
            viol, score, valid = _compute_violations_and_score(new_pos, new_radii, w, h)
            
            if valid:
                current_pos = new_pos
                current_radii = new_radii
            else:
                # If not valid, try to repair by scaling down radii
                # and accepting if the score is better than current
                current_viol, current_score, _ = _compute_violations_and_score(current_pos, current_radii, w, h)
                
                # Try scaling down
                for scale in [0.95, 0.9, 0.8, 0.7]:
                    rad_new = new_radii * scale
                    viol_new, score_new, valid_new = _compute_violations_and_score(new_pos, rad_new, w, h)
                    if valid_new and score_new > current_score:
                        current_pos = new_pos
                        current_radii = rad_new
                        break
        except Exception:
            pass
    
    return current_pos, current_radii

def _pack_for_aspect(w, h, start_time, deadline):
    best_score = 0
    best_pos = None
    best_radii = None
    
    max_r = min(w, h) / 2.0
    bounds = []
    for i in range(N):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0, max_r))
        
    initializations = []
    
    # 1. Hexagonal lattice start
    if time.time() < deadline - 15:
        pos_hex, rad_hex = _hexagonal_start(w, h, N, seed=42)
        x_hex = np.concatenate([pos_hex.flatten(), rad_hex])
        initializations.append(('hex', x_hex))
        
        # Repair initial hex to ensure validity for baseline
        pos_hex, rad_hex, score_hex = _repair(pos_hex, rad_hex, w, h)
        if score_hex > best_score:
            best_score = score_hex
            best_pos = pos_hex.copy()
            best_radii = rad_hex.copy()
            
    # 2. Random restarts
    if time.time() < deadline - 10:
        for seed in [123, 456, 789, 1011, 2023, 3456, 7890, 1234]:
            if time.time() > deadline - 5:
                break
            pos_rand, rad_rand = _random_start(w, h, N, seed=seed)
            x_rand = np.concatenate([pos_rand.flatten(), rad_rand])
            initializations.append(('rand', x_rand))
            
    for idx, (name, x0) in enumerate(initializations):
        if time.time() > deadline - 10:
            break
            
        try:
            # First pass: coarse optimization
            result = minimize(_objective, x0, args=(w, h, 1000.0, 0.01), 
                              method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 400, 'ftol': 1e-10, 'maxls': 50})
            
            x_opt = result.x
            pos_opt = x_opt[:N*2].reshape(N, 2)
            rad_opt = x_opt[N*2:]
            rad_opt = np.maximum(rad_opt, 0)
            
            # Repair to ensure we have a valid candidate
            pos_opt, rad_opt, score_opt = _repair(pos_opt, rad_opt, w, h)
            
            # Polish phase
            if time.time() < deadline - 5:
                pos_opt, rad_opt = _polish(pos_opt, rad_opt, w, h, start_time, deadline)
                # Final repair after polish
                pos_opt, rad_opt, score_opt = _repair(pos_opt, rad_opt, w, h)
                
            if score_opt > best_score:
                best_score = score_opt
                best_pos = pos_opt.copy()
                best_radii = rad_opt.copy()
                
        except Exception:
            continue
            
    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    best_global_score = 0
    best_global_result = None
    
    np.random.seed(42)
    
    # Try a range of aspect ratios
    # Square (1.0) is often optimal for N=21? Or slightly rectangular.
    # Let's focus on 0.8 to 1.25
    aspect_ratios = np.linspace(0.8, 1.25, 10)
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        pos, radii, score = _pack_for_aspect(w, h, start_time, deadline)
        
        if pos is not None and score > best_global_score:
            # Double check validity
            viol, s, valid = _compute_violations_and_score(pos, radii, w, h)
            if valid:
                best_global_score = score
                best_global_result = np.column_stack([pos, radii])
    
    # Fallback if no valid result found (should not happen with _repair)
    if best_global_result is None:
        w = 1.0
        h = 1.0
        pos, radii, score = _hexagonal_start(w, h, N, seed=42)
        pos, radii, score = _repair(pos, radii, w, h)
        best_global_result = np.column_stack([pos, radii])
        
    return best_global_result
