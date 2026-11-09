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

def _angles_to_poly(angles):
    """Convert angles to polygon on unit circle."""
    x = np.cos(angles)
    y = np.sin(angles)
    return np.stack([x, y], axis=1)

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
    def _objective_soft(angle_params, temp=0.05):
        angles = base_angles + angle_params
        angles = np.sort(angles)
        angles = angles - np.min(angles)
        span = angles[-1] - angles[0]
        if span > 2.0 * np.pi + 0.1 or span < 2.0 * np.pi - 0.1:
            return 1.0
        poly = _angles_to_poly(angles)
        s = _soft_min_vertex_score(poly, temperature=temp)
        return -s if s > 0 else 1.0
    
    # Try several random starts with decreasing perturbation scale
    n_starts = 8
    for start_idx in range(n_starts):
        if time.time() > deadline - 180:
            break
        
        if start_idx == 0:
            x0 = np.zeros(n)
        else:
            scale = 0.15 * (n_starts - start_idx) / n_starts
            x0 = rng.normal(0, scale, size=n)
        
        try:
            result = minimize(
                _objective_soft,
                x0,
                args=(0.05,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 40}
            )
            angles = base_angles + result.x
            angles = np.sort(angles)
            angles = angles - np.min(angles)
            poly = _angles_to_poly(angles)
            score = _compute_min_vertex_score(poly)
            if score > best_score + 1e-9:
                best_score = score
                best_poly = poly.copy()
        except Exception:
            pass
    
    if time.time() > deadline - 120:
        return best_poly

    # Phase 2: Fine search with tighter temperature
    # Extract current best angles
    best_angles_current = np.arctan2(best_poly[:, 1], best_poly[:, 0])
    order = np.argsort(best_angles_current)
    best_angles_sorted = best_angles_current[order]
    best_angles_sorted = best_angles_sorted - best_angles_sorted[0]
    angle_offset = best_angles_sorted - base_angles
    
    def _objective_fine(angle_params, temp=0.02):
        angles = base_angles + angle_offset + angle_params
        angles = np.sort(angles)
        angles = angles - np.min(angles)
        span = angles[-1] - angles[0]
        if span > 2.0 * np.pi + 0.05 or span < 2.0 * np.pi - 0.05:
            return 1.0
        poly = _angles_to_poly(angles)
        s = _soft_min_vertex_score(poly, temperature=temp)
        return -s if s > 0 else 1.0
    
    try:
        result = minimize(
            _objective_fine,
            np.zeros(n),
            args=(0.02,),
            method='L-BFGS-B',
            options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        angles = base_angles + angle_offset + result.x
        angles = np.sort(angles)
        angles = angles - np.min(angles)
        poly = _angles_to_poly(angles)
        score = _compute_min_vertex_score(poly)
        if score > best_score + 1e-9:
            best_score = score
            best_poly = poly.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 60:
        return best_poly

    # Phase 3: High-precision polish with hard score using Nelder-Mead
    best_angles_current = np.arctan2(best_poly[:, 1], best_poly[:, 0])
    order = np.argsort(best_angles_current)
    best_angles_sorted = best_angles_current[order]
    best_angles_sorted = best_angles_sorted - best_angles_sorted[0]
    angle_offset = best_angles_sorted - base_angles
    
    def _objective_hard(angle_params):
        angles = base_angles + angle_offset + angle_params
        angles = np.sort(angles)
        angles = angles - np.min(angles)
        span = angles[-1] - angles[0]
        if span > 2.0 * np.pi + 0.02 or span < 2.0 * np.pi - 0.02:
            return 1.0
        poly = _angles_to_poly(angles)
        s = _compute_min_vertex_score(poly)
        return -s if s > 0 else 1.0
    
    try:
        result = minimize(
            _objective_hard,
            np.zeros(n),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
        )
        angles = base_angles + angle_offset + result.x
        angles = np.sort(angles)
        angles = angles - np.min(angles)
        poly = _angles_to_poly(angles)
        score = _compute_min_vertex_score(poly)
        if score > best_score + 1e-9:
            best_score = score
            best_poly = poly.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 30:
        return best_poly

    # Phase 4: Final coordinate descent polish
    current_poly = best_poly.copy()
    current_score = best_score
    step_size = 0.001
    no_improve = 0
    
    while time.time() < deadline - 10 and no_improve < 20:
        improved = False
        for i in range(20):
            if time.time() > deadline - 10:
                break
            old_v = current_poly[i].copy()
            best_local = current_score
            best_new = None
            
            for mult in [1.0, 0.5, 0.25]:
                ss = step_size * mult
                for dx in [-ss, ss]:
                    for dy in [-ss, ss]:
                        new_poly = current_poly.copy()
                        new_poly[i] = old_v + np.array([dx, dy])
                        s = _compute_min_vertex_score(new_poly)
                        if s > best_local + 1e-12:
                            best_local = s
                            best_new = new_poly
                for d in [-ss, ss]:
                    for axis in [0, 1]:
                        new_poly = current_poly.copy()
                        new_poly[i] = old_v
                        new_poly[i, axis] += d
                        s = _compute_min_vertex_score(new_poly)
                        if s > best_local + 1e-12:
                            best_local = s
                            best_new = new_poly
            
            if best_new is not None:
                current_poly = best_new
                current_score = best_local
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_poly = current_poly.copy()
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            step_size *= 0.5
            if step_size < 1e-12:
                break

    return best_poly
