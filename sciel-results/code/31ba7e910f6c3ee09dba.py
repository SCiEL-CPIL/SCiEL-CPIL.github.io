import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def get_points_from_weights(weights):
    """
    weights: array of 20 positive numbers.
    Returns (20, 2) array of points.
    """
    w_sum = np.sum(weights)
    if w_sum <= 0:
        w_sum = 1e-6
    angles = np.cumsum(weights / w_sum * 2 * np.pi)
    angles = angles % (2 * np.pi)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))

def params_to_weights(params):
    """
    Convert 19 parameters to 20 positive weights.
    """
    w = params**2 + 0.01
    w = w / np.sum(w)
    return w * 2 * np.pi

def get_points(params):
    w = params_to_weights(params)
    angles = np.cumsum(w)
    angles = angles % (2 * np.pi)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))

def vertex_score(v_idx, points):
    dists = np.linalg.norm(points - points[v_idx], axis=1)
    dists[v_idx] = np.inf
    sorted_dists = np.sort(dists)
    if sorted_dists[0] < 0.05:
        return 0.0
    gaps = sorted_dists[1:] - sorted_dists[:-1]
    k = np.argmin(gaps)
    D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        d_sel = sorted_dists[0:4]
    elif k == 17:
        d_sel = sorted_dists[15:19]
    else:
        d_sel = sorted_dists[k-1:k+3]
    ratios = np.maximum(d_sel / D, D / d_sel)
    raw = np.mean(ratios)
    score = min(1.0, 1.0 / raw)
    return score

def soft_min(scores, beta=20.0):
    """
    Smooth minimum using log-sum-exp.
    Returns approximately min(scores) when beta is large.
    """
    scores = np.asarray(scores, dtype=float)
    m = np.min(scores)
    return m - np.log(np.sum(np.exp(-beta * (scores - m)))) / beta

def min_vertex_score(params):
    points = get_points(params)
    scores = np.array([vertex_score(i, points) for i in range(20)])
    return -soft_min(scores, beta=20.0)

def exact_min_vertex_score(params):
    """
    Compute the exact minimum vertex score (nonsmooth).
    """
    points = get_points(params)
    scores = np.array([vertex_score(i, points) for i in range(20)])
    return -np.min(scores)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590
    
    n = 20
    # Initial guess: regular polygon
    x0 = np.ones(20)
    
    best_params = x0.copy()
    best_score = -exact_min_vertex_score(x0)
    
    # Phase 1: Differential Evolution with soft-min objective
    if time.time() < deadline - 200:
        try:
            bounds = [(0.1, 10.0) for _ in range(20)]
            res_de = differential_evolution(
                min_vertex_score,
                bounds,
                maxiter=150,
                popsize=15,
                tol=1e-8,
                update_interval=1,
                seed=42
            )
            if -exact_min_vertex_score(res_de.x) > best_score:
                best_score = -exact_min_vertex_score(res_de.x)
                best_params = res_de.x.copy()
        except Exception:
            pass

    # Phase 2: Local Refinement with L-BFGS-B on soft-min
    rng = np.random.default_rng(42)
    
    restarts = [best_params.copy()]
    for i in range(15):
        if time.time() > deadline - 50:
            break
        # Random restarts
        x_rand = rng.uniform(0.5, 2.0, size=20)
        restarts.append(x_rand)
        # Also try perturbations of the best
        x_pert = best_params + rng.normal(0, 0.1, size=20)
        x_pert = np.clip(x_pert, 0.1, 10.0)
        restarts.append(x_pert)
        
    for x_init in restarts:
        if time.time() > deadline - 20:
            break
        try:
            # First, refine with soft-min
            res_lm = minimize(
                min_vertex_score,
                x_init,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if -exact_min_vertex_score(res_lm.x) > best_score:
                best_score = -exact_min_vertex_score(res_lm.x)
                best_params = res_lm.x.copy()
        except Exception:
            pass

    # Phase 3: Final high-precision polish on the exact objective
    if time.time() < deadline - 10:
        try:
            res_final = minimize(
                exact_min_vertex_score,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if -exact_min_vertex_score(res_final.x) > best_score:
                best_score = -exact_min_vertex_score(res_final.x)
                best_params = res_final.x.copy()
        except Exception:
            pass

    # Ensure validity: check that the polygon is strictly convex
    points = get_points(best_params)
    # Check for repeated points
    for i in range(20):
        for j in range(i+1, 20):
            if np.linalg.norm(points[i] - points[j]) < 1e-6:
                # Fallback to regular polygon
                angles = np.linspace(0, 2*np.pi, 20, endpoint=False)
                points = np.column_stack((np.cos(angles), np.sin(angles)))
                break
    
    # Normalize: subtract centroid, divide by mean distance
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist > 0.01:
        points = points / mean_dist
    
    return points
