import numpy as np
import time
from scipy.optimize import minimize
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

    min_score = 1.0
    for i in range(n):
        d = dists[i, :]
        d = d[np.isfinite(d)]
        d_sorted = np.sort(d)
        score = _vertex_score(d_sorted)
        if score < min_score:
            min_score = score
            if min_score == 0.0:
                return 0.0
    return min_score

def _soft_min_vertex_score(polygon, temp=1.0):
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
    deadline = start_time + 620.0

    best_poly = _regular_polygon()
    best_score = _compute_score(best_poly)
    
    n = 20
    base_theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # Parameterization: r_i (radial) and delta_theta_i (angle deviation from regular)
    # This keeps angles close to regular, helping convexity
    init_r = np.ones(n)
    init_dtheta = np.zeros(n)
    
    def _params_to_poly(params):
        r = params[:n]
        dtheta = params[n:]
        theta = base_theta + dtheta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack([x, y], axis=1)
    
    def _objective_exact(flat_params):
        poly = flat_params.reshape(n, 2)
        score = _compute_score(poly)
        return -score
    
    def _objective_smooth(flat_params, temp=1.0):
        poly = flat_params.reshape(n, 2)
        score = _soft_min_vertex_score(poly, temp)
        return -score
    
    current_params = best_poly.flatten()
    
    # Phase 1: Smooth optimization in Cartesian space using L-BFGS-B
    # We optimize the negative soft-min score.
    temps = [2.0, 1.0, 0.5, 0.25, 0.1]
    
    for temp in temps:
        if time.time() > deadline - 150:
            break
        
        def obj_smooth(flat_params):
            return _objective_smooth(flat_params, temp)
        
        try:
            res = minimize(
                obj_smooth,
                current_params,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            cand = res.x.reshape(n, 2)
            exact = _compute_score(cand)
            if exact > best_score:
                best_score = exact
                best_poly = cand
                current_params = cand.flatten()
        except Exception:
            pass

    if time.time() > deadline - 100:
        return best_poly

    # Phase 2: Nelder-Mead on exact metric for final polish
    # This handles the non-smoothness of the 'k' index changes better.
    try:
        res = minimize(
            _objective_exact,
            current_params,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        score = -res.fun
        if score > best_score:
            best_score = score
            best_poly = res.x.reshape(n, 2)
    except Exception:
        pass

    if time.time() > deadline - 50:
        return best_poly

    # Phase 3: BFGS polish with numerical gradients
    try:
        res = minimize(
            _objective_exact,
            best_poly.flatten(),
            method='BFGS',
            options={'maxiter': 200, 'gtol': 1e-10, 'disp': False}
        )
        score = -res.fun
        if score > best_score:
            best_score = score
            best_poly = res.x.reshape(n, 2)
    except Exception:
        pass

    # Final check
    final_score = _compute_score(best_poly)
    if final_score > best_score:
        best_score = final_score

    return best_poly
