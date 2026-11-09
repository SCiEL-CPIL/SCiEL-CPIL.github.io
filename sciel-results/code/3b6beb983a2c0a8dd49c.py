import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def params_to_weights(params):
    """
    Convert 19 parameters to 20 positive weights.
    The 20th weight is fixed to 1.0 to remove scale redundancy.
    """
    w_full = np.append(params**2 + 0.01, 1.0)
    w_full = w_full / np.sum(w_full)
    return w_full * 2 * np.pi

def get_points(params):
    """
    params: 19 values. Returns (20, 2) array of points on unit circle.
    """
    w = params_to_weights(params)
    angles = np.cumsum(w)
    angles = angles % (2 * np.pi)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))

def vertex_scores_all(points):
    """
    Compute vertex scores for all 20 vertices. Returns array of 20 scores.
    """
    n = 20
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    scores = np.zeros(n)
    for i in range(n):
        d = dists[i, :].copy()
        d[i] = np.inf
        sorted_d = np.sort(d)
        if sorted_d[0] < 0.05:
            scores[i] = 0.0
            continue
        gaps = sorted_d[1:] - sorted_d[:-1]
        k = np.argmin(gaps)
        D = (sorted_d[k] + sorted_d[k+1]) / 2.0
        if D < 0.1:
            scores[i] = 0.0
            continue
        if k == 0:
            d_sel = sorted_d[0:4]
        elif k == 17:
            d_sel = sorted_d[15:19]
        else:
            d_sel = sorted_d[k-1:k+3]
        ratios = np.maximum(d_sel / D, D / d_sel)
        raw = np.mean(ratios)
        scores[i] = min(1.0, 1.0 / raw)
    return scores

def soft_min(scores, beta=30.0):
    """
    Smooth minimum using log-sum-exp.
    """
    scores = np.asarray(scores, dtype=float)
    m = np.min(scores)
    return m - np.log(np.sum(np.exp(-beta * (scores - m)))) / beta

def min_vertex_score(params):
    points = get_points(params)
    scores = vertex_scores_all(points)
    return -soft_min(scores, beta=30.0)

def exact_min_vertex_score(params):
    points = get_points(params)
    scores = vertex_scores_all(points)
    return -np.min(scores)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590
    
    # Initial guess: regular polygon -> all 19 free params equal
    x0 = np.ones(19)
    
    best_params = x0.copy()
    best_score = -exact_min_vertex_score(x0)
    
    # Phase 1: Differential Evolution with soft-min objective
    if time.time() < deadline - 200:
        try:
            bounds = [(0.1, 10.0) for _ in range(19)]
            res_de = differential_evolution(
                min_vertex_score,
                bounds,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                update_interval=1,
                seed=42
            )
            s = -exact_min_vertex_score(res_de.x)
            if s > best_score:
                best_score = s
                best_params = res_de.x.copy()
        except Exception:
            pass

    # Phase 2: Local Refinement with L-BFGS-B on soft-min
    rng = np.random.default_rng(42)
    
    restarts = [best_params.copy()]
    for i in range(20):
        if time.time() > deadline - 50:
            break
        x_rand = rng.uniform(0.5, 2.0, size=19)
        restarts.append(x_rand)
        x_pert = best_params + rng.normal(0, 0.1, size=19)
        x_pert = np.clip(x_pert, 0.1, 10.0)
        restarts.append(x_pert)
        
    for x_init in restarts:
        if time.time() > deadline - 20:
            break
        try:
            res_lm = minimize(
                min_vertex_score,
                x_init,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            s = -exact_min_vertex_score(res_lm.x)
            if s > best_score:
                best_score = s
                best_params = res_lm.x.copy()
        except Exception:
            pass

    # Phase 3: Multiple Nelder-Mead restarts on exact objective
    if time.time() < deadline - 10:
        nm_starts = [best_params.copy()]
        for i in range(10):
            if time.time() > deadline - 10:
                break
            x_pert = best_params + rng.normal(0, 0.05, size=19)
            x_pert = np.clip(x_pert, 0.1, 10.0)
            nm_starts.append(x_pert)
        
        for x_init in nm_starts:
            if time.time() > deadline - 5:
                break
            try:
                res_nm = minimize(
                    exact_min_vertex_score,
                    x_init,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                s = -exact_min_vertex_score(res_nm.x)
                if s > best_score:
                    best_score = s
                    best_params = res_nm.x.copy()
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
