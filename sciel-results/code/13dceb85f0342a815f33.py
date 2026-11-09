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
    # Find first index minimizing the difference
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

    # Check convexity (Vectorized)
    # e1[i] = p[i+1] - p[i]
    # e2[i] = p[i+2] - p[i+1]
    p_next = np.roll(p, -1, axis=0)
    p_next2 = np.roll(p, -2, axis=0)
    e1 = p_next - p
    e2 = p_next2 - p_next
    cross_products = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    
    if np.any(np.abs(cross_products) < 1e-10):
        return 0.0
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return 0.0

    # Check total turning (Vectorized angles)
    # Angles of edges e1
    ang1 = np.arctan2(e1[:, 1], e1[:, 0])
    # Angles of edges e2 (which are e1 shifted by 1)
    ang2 = np.arctan2(e2[:, 1], e2[:, 0])
    
    turns = ang2 - ang1
    # Normalize turns to [-pi, pi]
    turns = (turns + np.pi) % (2 * np.pi) - np.pi
    total_turn = np.sum(turns)
    
    if not np.isclose(abs(total_turn), 2 * np.pi, atol=1e-6):
        return 0.0

    # Distances
    # p is (20, 2). We need pairwise distances.
    # diff[i, j] = p[i] - p[j]
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Mask diagonal
    idx = np.arange(n)
    dists[idx, idx] = np.inf
    
    min_score = 1.0
    for i in range(n):
        d = dists[i, :]
        d = d[np.isfinite(d)] # Remove inf
        d_sorted = np.sort(d)
        score = _vertex_score(d_sorted)
        if score < min_score:
            min_score = score
            if min_score == 0.0:
                return 0.0
    return min_score

def _soft_min_vertex_score(polygon, temp=1.0):
    """Soft-min surrogate using log-sum-exp for smooth gradients."""
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

    p_next = np.roll(p, -1, axis=0)
    p_next2 = np.roll(p, -2, axis=0)
    e1 = p_next - p
    e2 = p_next2 - p_next
    cross_products = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    
    if np.any(np.abs(cross_products) < 1e-10):
        return 0.0
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return 0.0

    ang1 = np.arctan2(e1[:, 1], e1[:, 0])
    ang2 = np.arctan2(e2[:, 1], e2[:, 0])
    turns = ang2 - ang1
    turns = (turns + np.pi) % (2 * np.pi) - np.pi
    total_turn = np.sum(turns)
    if not np.isclose(abs(total_turn), 2 * np.pi, atol=1e-6):
        return 0.0

    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    idx = np.arange(n)
    dists[idx, idx] = np.inf
    
    scores = np.zeros(n)
    for i in range(n):
        d = dists[i, :]
        d = d[np.isfinite(d)]
        d_sorted = np.sort(d)
        scores[i] = _vertex_score(d_sorted)
    
    if np.any(scores <= 0):
        return 0.0
    
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
    deadline = start_time + 590.0  # Safety margin
    
    best_poly = _regular_polygon()
    best_score = _compute_score(best_poly)
    
    n = 20
    base_theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # Parameterization: r_i (radial) and delta_theta_i (angle deviation)
    # Using delta_theta keeps angles close to regular, aiding convexity
    init_r = np.ones(n)
    init_dtheta = np.zeros(n)
    
    def _params_to_poly(params):
        r = params[:n]
        dtheta = params[n:]
        theta = base_theta + dtheta
        # Ensure r is positive
        r = np.maximum(r, 0.1) 
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack([x, y], axis=1)
    
    def _objective_exact(params):
        poly = _params_to_poly(params)
        score = _compute_score(poly)
        return -score
    
    def _objective_smooth(params, temp=1.0):
        poly = _params_to_poly(params)
        score = _soft_min_vertex_score(poly, temp)
        return -score
    
    # Bounds: r in [0.5, 2.0], dtheta in [-pi/10, pi/10]
    bounds = []
    for _ in range(n):
        bounds.append((0.5, 2.0))
    for _ in range(n):
        bounds.append((-np.pi/10, np.pi/10))
    
    current_params = np.concatenate([init_r, init_dtheta])
    
    # Phase 1: DE on smooth surrogate with decreasing temperature
    # This helps navigate the landscape without getting stuck on non-differentiable points
    temps = [5.0, 2.0, 1.0, 0.5, 0.25, 0.1]
    for temp in temps:
        if time.time() > deadline - 250:
            break
        
        def obj_smooth(params):
            return _objective_smooth(params, temp)
        
        try:
            result_de = differential_evolution(
                obj_smooth,
                bounds,
                x0=current_params,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                seed=42 + int(temp * 10),
                polish=False,
                maxfev=4000
            )
            if result_de.fun < 0:
                candidate = _params_to_poly(result_de.x)
                exact_score = _compute_score(candidate)
                if exact_score > best_score:
                    best_score = exact_score
                    best_poly = candidate
                    current_params = result_de.x
        except Exception:
            pass
    
    if time.time() > deadline - 180:
        return best_poly
    
    # Phase 2: DE on exact metric
    # The exact metric has non-smooth minima (due to sorting and min), so DE is better than gradient methods here
    for seed in [123, 456, 789, 101, 202, 303, 404]:
        if time.time() > deadline - 100:
            break
        try:
            result_de = differential_evolution(
                _objective_exact,
                bounds,
                x0=current_params,
                maxiter=50,
                popsize=12,
                tol=1e-5,
                seed=seed,
                polish=False,
                maxfev=2000
            )
            score = -result_de.fun
            if score > best_score:
                best_score = score
                best_poly = _params_to_poly(result_de.x)
                current_params = result_de.x
        except Exception:
            pass
    
    if time.time() > deadline - 60:
        return best_poly
    
    # Phase 3: Local refinement using Nelder-Mead in Cartesian space
    # Nelder-Mead is derivative-free and robust for the final polish
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
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True, 'maxls': 200}
        )
        score = -result.fun
        if score > best_score:
            best_score = score
            best_poly = result.x.reshape(n, 2)
    except Exception:
        pass
    
    # Final high-precision polish
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _objective_cart,
                best_poly.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            score = -result.fun
            if score > best_score:
                best_score = score
                best_poly = result.x.reshape(n, 2)
        except Exception:
            pass
    
    return best_poly
