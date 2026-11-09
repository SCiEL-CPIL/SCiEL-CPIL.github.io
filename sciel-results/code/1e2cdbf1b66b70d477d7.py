import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _vertex_score(distances_sorted):
    d = distances_sorted
    n = len(d)
    if n != 19:
        return 0.0
    if np.any(d < 0.05):
        return 0.0
    diffs = d[1:] - d[:-1]
    min_diff = np.min(diffs)
    # Find first index minimizing the gap
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
    # Check for repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(p[i] - p[j]) < 1e-10:
                return False
    cross_products = np.zeros(n)
    for i in range(n):
        e1 = p[(i+1) % n] - p[i]
        e2 = p[(i+2) % n] - p[(i+1) % n]
        cross_products[i] = e1[0] * e2[1] - e1[1] * e2[0]
    if np.any(np.abs(cross_products) < 1e-12):
        return False
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return False
    # Check total turning
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

def _superellipse_polygon(p=4.0, angle_offsets=None):
    """
    Generate a superellipse (Lamé curve) polygon.
    x^p + y^p = 1 (scaled)
    Parametric: x = cos(t)^(2/p), y = sin(t)^(2/p)
    """
    n = 20
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    if angle_offsets is not None:
        t = t + angle_offsets
    x = np.sign(np.cos(t)) * (np.abs(np.cos(t)) ** (2.0 / p))
    y = np.sign(np.sin(t)) * (np.abs(np.sin(t)) ** (2.0 / p))
    return np.stack([x, y], axis=1)

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
        
    # Force CCW (positive cross products)
    neg_cross = np.maximum(0, -cross_products)
    penalty = np.sum(neg_cross**2)
    
    return weight * penalty

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 615.0
    
    n = 20
    best_poly = _regular_polygon()
    best_score = _compute_min_vertex_score(best_poly)
    
    # Phase 1: Global search with differential evolution on superellipse params
    # Parameters: [log(p), angle_offset_0, ..., angle_offset_19]
    # p = exp(log_p) controls superellipse shape
    
    def _de_objective(params):
        log_p = params[0]
        p_val = np.exp(log_p)
        angle_offsets = params[1:]
        # Keep offsets small to maintain convexity
        if np.any(np.abs(angle_offsets) > 0.5):
            return 10.0
        poly = _superellipse_polygon(p_val, angle_offsets)
        s = _soft_min_vertex_score(poly, temperature=0.05)
        return -s if s > 0 else 10.0
    
    bounds_de = [(-1.0, 2.0)] + [(-0.3, 0.3)] * n
    try:
        if time.time() < deadline - 300:
            de_result = differential_evolution(
                _de_objective,
                bounds_de,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=True
            )
            log_p_opt = de_result.x[0]
            p_opt = np.exp(log_p_opt)
            angle_offsets_opt = de_result.x[1:]
            poly = _superellipse_polygon(p_opt, angle_offsets_opt)
            score = _compute_min_vertex_score(poly)
            if score > best_score + 1e-9:
                best_score = score
                best_poly = poly.copy()
    except Exception:
        pass
    
    # Phase 2: Local refinement with L-BFGS-B on superellipse params
    # Start from best found so far
    
    # Extract current superellipse params from best_poly
    centroid = np.mean(best_poly, axis=0)
    p_centered = best_poly - centroid
    current_angles = np.arctan2(p_centered[:, 1], p_centered[:, 0])
    current_radii = np.sqrt(np.sum(p_centered**2, axis=1))
    
    # Estimate p from radial variation
    # For superellipse, r(θ) = (|cos(θ)|^(2/p) + |sin(θ)|^(2/p))^(-1/2)
    # Fit p by minimizing residual
    def _fit_p(p_val):
        t = current_angles
        x = np.sign(np.cos(t)) * (np.abs(np.cos(t)) ** (2.0 / p_val))
        y = np.sign(np.sin(t)) * (np.abs(np.sin(t)) ** (2.0 / p_val))
        # Scale to match mean radius
        r_fit = np.sqrt(x**2 + y**2)
        if np.mean(r_fit) < 0.01:
            return 100.0
        r_fit_scaled = r_fit / np.mean(r_fit) * np.mean(current_radii)
        return np.sum((r_fit_scaled - current_radii)**2)
    
    # Try to fit p
    best_p_fit = 4.0
    for p_test in np.linspace(2.0, 10.0, 41):
        res = _fit_p(p_test)
        if res < _fit_p(best_p_fit):
            best_p_fit = p_test
    
    # Angle offsets: difference from regular polygon angles
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    angle_diff = current_angles - base_angles
    # Unwrap and center
    angle_diff = np.unwrap(angle_diff)
    angle_diff -= np.mean(angle_diff)
    
    # Local optimization on [log_p, angle_offsets]
    def _local_objective(params):
        log_p = params[0]
        p_val = np.exp(log_p)
        angle_offsets = params[1:]
        if np.any(np.abs(angle_offsets) > 0.5):
            return 10.0
        poly = _superellipse_polygon(p_val, angle_offsets)
        s = _soft_min_vertex_score(poly, temperature=0.03)
        return -s if s > 0 else 10.0
    
    x0_local = np.concatenate([[np.log(best_p_fit)], angle_diff])
    try:
        if time.time() < deadline - 150:
            result = minimize(
                _local_objective,
                x0_local,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            log_p_opt = result.x[0]
            p_opt = np.exp(log_p_opt)
            angle_offsets_opt = result.x[1:]
            poly = _superellipse_polygon(p_opt, angle_offsets_opt)
            score = _compute_min_vertex_score(poly)
            if score > best_score + 1e-9:
                best_score = score
                best_poly = poly.copy()
    except Exception:
        pass
    
    # Phase 3: Coordinate-level refinement with soft surrogate
    # Directly optimize all 40 coordinates
    
    def _objective_coord_soft(params, temp=0.02):
        p = params.reshape(n, 2)
        if not np.all(np.isfinite(p)):
            return 100.0
        centroid = np.mean(p, axis=0)
        p_centered = p - centroid
        dists = np.sqrt(np.sum(p_centered**2, axis=1))
        mean_dist = np.mean(dists)
        if mean_dist <= 0.01:
            return 100.0
        
        # Check convexity
        cross_products = np.zeros(n)
        for i in range(n):
            e1 = p_centered[(i+1) % n] - p_centered[i]
            e2 = p_centered[(i+2) % n] - p_centered[(i+1) % n]
            cross_products[i] = e1[0] * e2[1] - e1[1] * e2[0]
        if not np.all(cross_products > 1e-8):
            return 10.0 + np.sum(np.maximum(0, -cross_products)**2)
            
        s = _soft_min_vertex_score(p, temperature=temp)
        return -s if s > 0 else 10.0
    
    try:
        if time.time() < deadline - 80:
            result = minimize(
                _objective_coord_soft,
                best_poly.flatten(),
                args=(0.02,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            p_opt = result.x.reshape(n, 2)
            score = _compute_min_vertex_score(p_opt)
            if score > best_score + 1e-9:
                best_score = score
                best_poly = p_opt.copy()
    except Exception:
        pass
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 30:
        def _objective_final(params, temp=0.01):
            p = params.reshape(n, 2)
            if not np.all(np.isfinite(p)):
                return 100.0
            centroid = np.mean(p, axis=0)
            p_centered = p - centroid
            dists = np.sqrt(np.sum(p_centered**2, axis=1))
            mean_dist = np.mean(dists)
            if mean_dist <= 0.01:
                return 100.0
            
            cross_products = np.zeros(n)
            for i in range(n):
                e1 = p_centered[(i+1) % n] - p_centered[i]
                e2 = p_centered[(i+2) % n] - p_centered[(i+1) % n]
                cross_products[i] = e1[0] * e2[1] - e1[1] * e2[0]
            if not np.all(cross_products > 1e-8):
                return 10.0 + np.sum(np.maximum(0, -cross_products)**2)
                
            s = _soft_min_vertex_score(p, temperature=temp)
            return -s if s > 0 else 10.0
        
        try:
            result = minimize(
                _objective_final,
                best_poly.flatten(),
                args=(0.01,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            p_opt = result.x.reshape(n, 2)
            score = _compute_min_vertex_score(p_opt)
            if score > best_score + 1e-9:
                best_score = score
                best_poly = p_opt.copy()
        except Exception:
            pass
    
    return best_poly
