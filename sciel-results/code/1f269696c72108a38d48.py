import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import warnings
warnings.filterwarnings('ignore')

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

def _compute_score(polygon):
    n = len(polygon)
    if n != 20:
        return 0.0
    if not np.all(np.isfinite(polygon)):
        return 0.0
    
    # Normalization
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(p**2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    p = p / mean_dist

    # Check convexity
    cross_products = np.zeros(n)
    for i in range(n):
        e1 = p[(i+1) % n] - p[i]
        e2 = p[(i+2) % n] - p[(i+1) % n]
        cross_products[i] = e1[0] * e2[1] - e1[1] * e2[0]
    
    # Must be strictly convex: all cross products same sign and nonzero
    if np.any(np.abs(cross_products) < 1e-10):
        return 0.0
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return 0.0

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
        return 0.0

    # Distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    min_score = 1.0
    for i in range(n):
        d = np.delete(dists[i], i)
        d_sorted = np.sort(d)
        score = _vertex_score(d_sorted)
        if score < min_score:
            min_score = score
    return min_score

def _soft_min_vertex_score(polygon, temp=1.0):
    """Soft-min surrogate using log-sum-exp for smooth gradients."""
    n = len(polygon)
    if n != 20:
        return 0.0
    if not np.all(np.isfinite(polygon)):
        return 0.0
    
    # Normalization
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(p**2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    p = p / mean_dist

    # Check convexity
    cross_products = np.zeros(n)
    for i in range(n):
        e1 = p[(i+1) % n] - p[i]
        e2 = p[(i+2) % n] - p[(i+1) % n]
        cross_products[i] = e1[0] * e2[1] - e1[1] * e2[0]
    
    if np.any(np.abs(cross_products) < 1e-10):
        return 0.0
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return 0.0

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
        return 0.0

    # Distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    scores = np.zeros(n)
    for i in range(n):
        d = np.delete(dists[i], i)
        d_sorted = np.sort(d)
        scores[i] = _vertex_score(d_sorted)
    
    if np.any(scores <= 0):
        return 0.0
    
    # Log-sum-exp soft-min
    # soft_min = -log(sum(exp(-s_i/temp)))/temp
    neg_scores = -scores / temp
    max_neg = np.max(neg_scores)
    log_sum = np.log(np.sum(np.exp(neg_scores - max_neg))) + max_neg
    soft_min = -log_sum * temp
    return max(0.0, min(1.0, soft_min))

def _regular_polygon():
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0
    
    best_poly = _regular_polygon()
    best_score = _compute_score(best_poly)
    
    n = 20
    init_r = np.ones(n)
    init_theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    def _params_to_poly(params):
        r = params[:n]
        theta = params[n:]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack([x, y], axis=1)
    
    def _objective(params):
        poly = _params_to_poly(params)
        score = _compute_score(poly)
        return -score
    
    def _smooth_objective(params, temp=1.0):
        poly = _params_to_poly(params)
        score = _soft_min_vertex_score(poly, temp)
        return -score
    
    bounds = []
    for _ in range(n):
        bounds.append((0.3, 3.0))
    for _ in range(n):
        bounds.append((-np.pi, np.pi + 2 * 2 * np.pi / n))
    
    # Phase 1: DE on smooth surrogate with decreasing temperature
    temps = [5.0, 2.0, 1.0, 0.5]
    current_params = init_r.copy()
    current_params = np.concatenate([current_params, init_theta])
    
    for temp in temps:
        if time.time() > deadline - 180:
            break
        
        def obj_smooth(params):
            return _smooth_objective(params, temp)
        
        try:
            result_de = differential_evolution(
                obj_smooth,
                bounds,
                x0=current_params,
                maxiter=60,
                popsize=15,
                tol=1e-6,
                seed=42 + int(temp * 10),
                polish=False,
                maxfev=2000
            )
            if result_de.fun < 0:  # Valid
                candidate = _params_to_poly(result_de.x)
                exact_score = _compute_score(candidate)
                if exact_score > best_score:
                    best_score = exact_score
                    best_poly = candidate
                    current_params = result_de.x
        except Exception:
            pass
    
    if time.time() > deadline - 150:
        return best_poly
    
    # Phase 2: Local refinement on exact metric using Nelder-Mead
    def _objective_cart(flat_params):
        poly = flat_params.reshape(n, 2)
        score = _compute_score(poly)
        return -score
    
    x0 = best_poly.flatten()
    try:
        result = minimize(
            _objective_cart,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True, 'maxls': 100}
        )
        score = -result.fun
        if score > best_score:
            best_score = score
            best_poly = result.x.reshape(n, 2)
    except Exception:
        pass
    
    if time.time() > deadline - 90:
        return best_poly
    
    # Phase 3: More DE restarts with different seeds on exact metric
    for seed in [123, 456, 789]:
        if time.time() > deadline - 60:
            break
        try:
            result_de = differential_evolution(
                _objective,
                bounds,
                maxiter=30,
                popsize=12,
                tol=1e-5,
                seed=seed,
                polish=True,
                maxfev=1000
            )
            score = -result_de.fun
            if score > best_score:
                best_score = score
                best_poly = _params_to_poly(result_de.x)
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 30:
        try:
            x0 = best_poly.flatten()
            result = minimize(
                _objective_cart,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            score = -result.fun
            if score > best_score:
                best_score = score
                best_poly = result.x.reshape(n, 2)
        except Exception:
            pass
    
    return best_poly
