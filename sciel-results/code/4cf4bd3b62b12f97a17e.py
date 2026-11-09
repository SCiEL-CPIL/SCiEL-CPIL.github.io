import numpy as np
import time
from scipy.optimize import minimize

def _vertex_score(distances_sorted):
    d = distances_sorted
    n = len(d)
    if n != 19:
        return 0.0
    if np.any(d < 0.05):
        return 0.0
    diffs = d[1:] - d[:-1]
    min_diff = np.min(diffs)
    k = int(np.argmax(diffs == min_diff))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        four_d = d[0:4]
    elif k == 17:
        four_d = d[15:19]
    else:
        four_d = d[k-1:k+3]
    if np.any(four_d <= 0):
        return 0.0
    ratios = np.maximum(four_d / D, D / four_d)
    raw = np.mean(ratios)
    score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
    return float(score)

def _is_valid_convex(p):
    n = len(p)
    if not np.all(np.isfinite(p)):
        return False
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(p[i] - p[j]) < 1e-10:
                return False
    cross_products = np.zeros(n)
    for i in range(n):
        e1 = p[(i+1) % n] - p[i]
        e2 = p[(i+2) % n] - p[(i+1) % n]
        cross_products[i] = e1[0] * e2[1] - e1[1] * e2[0]
    if np.any(np.abs(cross_products) < 1e-14):
        return False
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return False
    total_turn = 0.0
    for i in range(n):
        v1 = p[(i+1) % n] - p[i]
        v2 = p[(i+2) % n] - p[(i+1) % n]
        ang1 = np.arctan2(v1[1], v1[0])
        ang2 = np.arctan2(v2[1], v2[0])
        turn = ang2 - ang1
        while turn > np.pi:
            turn -= 2 * np.pi
        while turn < -np.pi:
            turn += 2 * np.pi
        total_turn += turn
    if not np.isclose(abs(total_turn), 2 * np.pi, atol=1e-6):
        return False
    return True

def _compute_min_vertex_score(polygon):
    n = len(polygon)
    if n != 20:
        return 0.0
    if not np.all(np.isfinite(polygon)):
        return 0.0
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(p**2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    p = p / mean_dist
    if not _is_valid_convex(p):
        return 0.0
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    min_score = 1.0
    for i in range(n):
        d = np.delete(dists[i], i)
        d_sorted = np.sort(d)
        score = _vertex_score(d_sorted)
        min_score = min(min_score, score)
    return min_score

def _regular_polygon():
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)

def _soft_min_vertex_score(polygon, temperature=0.05):
    """Smooth surrogate for minimum vertex score using log-sum-exp."""
    n = len(polygon)
    if n != 20:
        return 0.0
    if not np.all(np.isfinite(polygon)):
        return 0.0
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(p**2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    p = p / mean_dist
    
    # Quick convexity check
    cross_products = np.zeros(n)
    for i in range(n):
        e1 = p[(i+1) % n] - p[i]
        e2 = p[(i+2) % n] - p[(i+1) % n]
        cross_products[i] = e1[0] * e2[1] - e1[1] * e2[0]
    if np.any(np.abs(cross_products) < 1e-12):
        return 0.0
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return 0.0
    
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    scores = np.zeros(n)
    for i in range(n):
        d = np.delete(dists[i], i)
        d_sorted = np.sort(d)
        scores[i] = _vertex_score(d_sorted)
    
    # Log-sum-exp for smooth minimum: min ≈ -T*log(sum(exp(-scores/T)))
    neg_scores = -scores
    max_neg = np.max(neg_scores)
    if max_neg >= 0:
        log_sum_exp_neg = max_neg + temperature * np.log(np.sum(np.exp((neg_scores - max_neg) / temperature)))
        smooth_max_neg = log_sum_exp_neg
        smooth_min_score = -smooth_max_neg
    else:
        smooth_min_score = -max_neg
    
    return max(0.0, smooth_min_score)

def _convexity_penalty(polygon, weight=1.0):
    """Penalty for non-convexity. Returns 0 if convex, positive otherwise."""
    n = len(polygon)
    if n != 20:
        return 1.0
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(p**2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 1.0
    p = p / mean_dist
    
    cross_products = np.zeros(n)
    for i in range(n):
        e1 = p[(i+1) % n] - p[i]
        e2 = p[(i+2) % n] - p[(i+1) % n]
        cross_products[i] = e1[0] * e2[1] - e1[1] * e2[0]
        
    # We want all cross products to have the same sign.
    # Let's assume CCW (positive cross products) for the regular polygon start.
    # If the polygon flips, the sign will change.
    # A robust penalty: sum of squares of cross products that have the "wrong" sign.
    # Or simply: we want min(cross_products) > 0 and max(cross_products) > 0? No.
    # If all are positive, it's convex (CCW). If all negative, convex (CW).
    # Let's force CCW.
    # Penalty = sum(max(0, -cross_products)^2)
    neg_cross = np.maximum(0, -cross_products)
    penalty = np.sum(neg_cross**2)
    
    # Also penalize if any cross product is too close to 0 (collinear)
    # But the score function handles "validity" somewhat.
    # The main issue is the sign flip.
    
    return weight * penalty

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 615.0
    
    n = 20
    best_poly = _regular_polygon()
    best_score = _compute_min_vertex_score(best_poly)
    
    if time.time() > deadline - 10:
        return best_poly

    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    rng = np.random.default_rng(123)
    
    # Phase 1: Coarse search with soft surrogate - multiple starts
    def _objective_soft_angle(angle_params, temp=0.05):
        angles = base_angles + angle_params
        angles = np.sort(angles)
        angles = angles - np.min(angles)
        span = angles[-1] - angles[0]
        if span > 2.0 * np.pi + 0.1 or span < 2.0 * np.pi - 0.1:
            return 1.0
        poly = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        s = _soft_min_vertex_score(poly, temperature=temp)
        return -s if s > 0 else 1.0
    
    # Try several random starts with decreasing perturbation scale
    n_starts = 5 # Reduced to save time for new phase
    for start_idx in range(n_starts):
        if time.time() > deadline - 250:
            break
        
        if start_idx == 0:
            x0 = np.zeros(n)
        else:
            scale = 0.15 * (n_starts - start_idx) / n_starts
            x0 = rng.normal(0, scale, size=n)
        
        try:
            result = minimize(
                _objective_soft_angle,
                x0,
                args=(0.05,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 40}
            )
            angles = base_angles + result.x
            angles = np.sort(angles)
            angles = angles - np.min(angles)
            poly = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            score = _compute_min_vertex_score(poly)
            if score > best_score + 1e-9:
                best_score = score
                best_poly = poly.copy()
        except Exception:
            pass
    
    if time.time() > deadline - 200:
        return best_poly

    # Phase 2: Fine search with tighter temperature (Cyclic)
    best_angles_current = np.arctan2(best_poly[:, 1], best_poly[:, 0])
    order = np.argsort(best_angles_current)
    best_angles_sorted = best_angles_current[order]
    best_angles_sorted = best_angles_sorted - best_angles_sorted[0]
    angle_offset = best_angles_sorted - base_angles
    
    def _objective_fine_angle(angle_params, temp=0.02):
        angles = base_angles + angle_offset + angle_params
        angles = np.sort(angles)
        angles = angles - np.min(angles)
        span = angles[-1] - angles[0]
        if span > 2.0 * np.pi + 0.05 or span < 2.0 * np.pi - 0.05:
            return 1.0
        poly = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        s = _soft_min_vertex_score(poly, temperature=temp)
        return -s if s > 0 else 1.0
    
    try:
        result = minimize(
            _objective_fine_angle,
            np.zeros(n),
            args=(0.02,),
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        angles = base_angles + angle_offset + result.x
        angles = np.sort(angles)
        angles = angles - np.min(angles)
        poly = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        score = _compute_min_vertex_score(poly)
        if score > best_score + 1e-9:
            best_score = score
            best_poly = poly.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 100:
        return best_poly

    # Phase 3: NEW - General Coordinate Search (Non-Cyclic)
    # Flatten the polygon to 40 parameters
    # We optimize the coordinates directly.
    # Objective: -SoftScore + ConvexityPenalty
    
    def _objective_coord(params, temp=0.05, pen_weight=1.0):
        p = params.reshape(n, 2)
        # Quick validity check to avoid expensive soft score if totally broken
        if not np.all(np.isfinite(p)):
            return 100.0
        centroid = np.mean(p, axis=0)
        p_centered = p - centroid
        dists = np.sqrt(np.sum(p_centered**2, axis=1))
        mean_dist = np.mean(dists)
        if mean_dist <= 0.01:
            return 100.0
        
        # Convexity penalty
        pen = _convexity_penalty(p, weight=pen_weight)
        if pen > 1.0:
            return 10.0 + pen
        
        s = _soft_min_vertex_score(p, temperature=temp)
        return -s + pen
    
    # Start from the best cyclic polygon
    x0 = best_poly.flatten()
    
    try:
        # Use BFGS (or L-BFGS-B) on the 40 coordinates
        # L-BFGS-B is good for larger problems, but 40 is small. BFGS is fine.
        # We don't have bounds, so BFGS is appropriate.
        result = minimize(
            _objective_coord,
            x0,
            args=(0.05, 1.0),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        
        p_opt = result.x.reshape(n, 2)
        score = _compute_min_vertex_score(p_opt)
        if score > best_score + 1e-9:
            best_score = score
            best_poly = p_opt.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 50:
        return best_poly

    # Phase 4: Polish with Hard Score and Nelder-Mead on Coordinates
    def _objective_hard_coord(params):
        p = params.reshape(n, 2)
        if not np.all(np.isfinite(p)):
            return 100.0
        centroid = np.mean(p, axis=0)
        p_centered = p - centroid
        dists = np.sqrt(np.sum(p_centered**2, axis=1))
        mean_dist = np.mean(dists)
        if mean_dist <= 0.01:
            return 100.0
        
        # Check convexity quickly
        cross_products = np.zeros(n)
        for i in range(n):
            e1 = p_centered[(i+1) % n] - p_centered[i]
            e2 = p_centered[(i+2) % n] - p_centered[(i+1) % n]
            cross_products[i] = e1[0] * e2[1] - e1[1] * e2[0]
        if not np.all(cross_products > 1e-8): # Assuming CCW
            # Try flipping? No, just penalize.
            return 10.0 + np.sum(np.maximum(0, -cross_products)**2)
            
        s = _compute_min_vertex_score(p)
        return -s if s > 0 else 10.0

    try:
        result = minimize(
            _objective_hard_coord,
            best_poly.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        p_opt = result.x.reshape(n, 2)
        score = _compute_min_vertex_score(p_opt)
        if score > best_score + 1e-9:
            best_score = score
            best_poly = p_opt.copy()
    except Exception:
        pass

    return best_poly
