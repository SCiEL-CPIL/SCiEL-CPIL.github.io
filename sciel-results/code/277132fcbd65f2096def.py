# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _vertex_score(distances_sorted):
    d = distances_sorted
    n = len(d)
    if n != 19:
        return 0.0
    if any(di < 0.05 for di in d):
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
    ratios = np.maximum(four_d / D, D / four_d)
    raw = np.mean(ratios)
    score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
    return float(score)

def _is_valid_convex(p):
    """Fast convexity check for normalized polygon."""
    n = len(p)
    if not np.all(np.isfinite(p)):
        return False
    
    # Check no repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.allclose(p[i], p[j], atol=1e-10):
                return False

    # Check strict convexity (all cross products same sign and non-zero)
    cross_products = []
    for i in range(n):
        v1 = p[(i+1) % n] - p[i]
        v2 = p[(i+2) % n] - p[(i+1) % n]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        cross_products.append(cross)
    cross_products = np.array(cross_products)
    if np.any(np.abs(cross_products) < 1e-12):
        return False
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return False
    
    # Check total turning angle is +/- 2*pi
    total_turn = 0.0
    for i in range(n):
        v1 = p[(i+1) % n] - p[i]
        v2 = p[(i+2) % n] - p[(i+1) % n]
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        turn = angle2 - angle1
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

    min_score = 1.0
    for i in range(n):
        diffs = p - p[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        dists = np.delete(dists, i)
        dists_sorted = np.sort(dists)
        score = _vertex_score(dists_sorted)
        min_score = min(min_score, score)
    return min_score

def _regular_polygon():
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.stack([x, y], axis=1)

def _soft_min_vertex_score(polygon, temperature=0.01):
    """
    Computes a smooth approximation of the minimum vertex score.
    Uses log-sum-exp trick for soft-min.
    """
    n = len(polygon)
    if not np.all(np.isfinite(polygon)):
        return -1.0
        
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(p**2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return -1.0
    p = p / mean_dist
    
    if not _is_valid_convex(p):
        return -1.0

    scores = np.zeros(n)
    for i in range(n):
        diffs = p - p[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        dists = np.delete(dists, i)
        dists_sorted = np.sort(dists)
        scores[i] = _vertex_score(dists_sorted)
        
    # Soft min: -log(sum(exp(-s/T))) / T
    # To avoid overflow, we shift by max score
    max_score = np.max(scores)
    if max_score == 0:
        return 0.0
    # If any score is 0, soft min will be close to 0
    # We want to maximize the min score, so we minimize -soft_min_score
    # soft_min = -1/T * log(sum(exp(-s/T)))
    # Let's just use a simple approximation: min + small penalty for variance? 
    # No, let's stick to the log-sum-exp for robustness.
    
    # Actually, standard soft-min is:
    # m = min(s)
    # soft_min = m - T * log( sum( exp((m - s_i) / T) ) ) / n ? No.
    # soft_min = -T * log( (1/n) * sum(exp(-s_i/T)) )
    
    # Let's compute -soft_min to minimize
    # -soft_min = T * log( (1/n) * sum(exp(-s_i/T)) )
    
    # For numerical stability:
    # exp(-s_i/T) can be huge if s_i is negative? No, scores are >= 0.
    # If s_i is large, exp(-s_i/T) is small.
    # Max of -s_i/T is -min(s_i)/T.
    
    neg_scores_over_T = -scores / temperature
    max_neg = np.max(neg_scores_over_T)
    if max_neg > 100: # Overflow protection
         return -1.0
        
    sum_exp = np.sum(np.exp(neg_scores_over_T - max_neg))
    log_sum = max_neg + np.log(sum_exp)
    soft_min_val = -temperature * (log_sum - np.log(n))
    
    return soft_min_val

def _optimize_with_scipy(initial_poly, max_time=580.0):
    """
    Optimize polygon vertices using scipy.optimize.minimize with a soft-min objective.
    """
    start_time = time.time()
    current_poly = initial_poly.copy()
    
    # Flatten the polygon for scipy
    x0 = current_poly.flatten()
    
    def objective(x):
        poly = x.reshape(20, 2)
        score = _soft_min_vertex_score(poly, temperature=0.02)
        # We want to maximize score, so minimize -score
        # If score is -1 (invalid), return large penalty
        if score < 0:
            return 1e6
        return -score
    
    # Use Nelder-Mead for initial exploration as it doesn't require gradients
    # Then switch to L-BFGS-B if possible? Nelder-Mead is robust for this kind of problem.
    
    best_poly = current_poly.copy()
    best_obj = objective(x0)
    
    # Run Nelder-Mead
    options = {
        'maxiter': 1000,
        'maxfev': 1000,
        'xatol': 1e-5,
        'fatol': 1e-5,
        'adaptive': True
    }
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options=options)
        if res.success or res.fun < best_obj:
            best_poly = res.x.reshape(20, 2)
            best_obj = res.fun
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > max_time:
        return best_poly, -best_obj if best_obj < 1e5 else 0.0
        
    # Refine with L-BFGS-B using numerical gradients (since we don't have analytic ones easily)
    # L-BFGS-B might get stuck in local minima if not careful, but Nelder-Mead usually finds a good basin.
    
    def objective_lf(x):
        poly = x.reshape(20, 2)
        score = _soft_min_vertex_score(poly, temperature=0.01)
        if score < 0:
            return 1e6
        return -score
        
    try:
        res2 = minimize(objective_lf, best_poly.flatten(), method='L-BFGS-B', 
                        options={'maxiter': 500, 'ftol': 1e-8})
        if res2.fun < best_obj:
            best_poly = res2.x.reshape(20, 2)
            best_obj = res2.fun
    except Exception:
        pass
        
    final_score = _compute_min_vertex_score(best_poly)
    return best_poly, final_score

def _try_radial_modulation():
    """
    Try polygons with modulated radii to improve distance uniformity.
    """
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    best_poly = _regular_polygon()
    best_score = _compute_min_vertex_score(best_poly)
    
    for a in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1]:
        for m in [1, 2, 3, 4, 5]:
            radii = 1.0 + a * np.cos(m * base_angles)
            x = radii * np.cos(base_angles)
            y = radii * np.sin(base_angles)
            poly = np.stack([x, y], axis=1)
            score = _compute_min_vertex_score(poly)
            if score > best_score:
                best_score = score
                best_poly = poly.copy()
                
    for a in [0.01, 0.02, 0.03, 0.05]:
        for m in [1, 2, 3]:
            angles = base_angles + a * np.sin(m * base_angles)
            order = np.argsort(angles)
            angles_sorted = angles[order]
            x = np.cos(angles_sorted)
            y = np.sin(angles_sorted)
            poly = np.stack([x, y], axis=1)
            score = _compute_min_vertex_score(poly)
            if score > best_score:
                best_score = score
                best_poly = poly.copy()
                
    return best_poly, best_score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0
    
    best_poly = _regular_polygon()
    best_score = _compute_min_vertex_score(best_poly)
    
    if time.time() > deadline:
        return best_poly
        
    mod_poly, mod_score = _try_radial_modulation()
    if mod_score > best_score:
        best_score = mod_score
        best_poly = mod_poly
        
    if time.time() > deadline:
        return best_poly
        
    # Optimize from the best initial guess
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        opt_poly, opt_score = _optimize_with_scipy(best_poly, max_time=remaining_time)
        if opt_score > best_score:
            best_score = opt_score
            best_poly = opt_poly
            
    return best_poly

# EVOLVE-BLOCK-END
