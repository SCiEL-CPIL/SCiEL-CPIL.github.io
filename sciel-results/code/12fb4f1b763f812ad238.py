import numpy as np
import time
from scipy.optimize import minimize

# Global variables
_ANGLES = None
_START_TIME = None
_DEADLINE = None

def _init():
    global _ANGLES, _START_TIME, _DEADLINE
    n = 20
    _ANGLES = np.linspace(0, 2 * np.pi, n, endpoint=False)
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 580.0

def _compute_scores(points):
    """
    Vectorized vertex-score computation.
    Returns an array of 20 scores in [0, 1].
    """
    n = len(points)
    # Pairwise distances: (n, n)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))

    # For each vertex i, sort distances to the other n-1 vertices.
    dists_self_inf = dists.copy()
    np.fill_diagonal(dists_self_inf, np.inf)
    sorted_d = np.sort(dists_self_inf, axis=1)  # (n, n), last col is inf
    d = sorted_d[:, :n - 1]  # (n, n-1)

    scores = np.zeros(n)

    # Check: any distance < 0.05 -> score 0
    has_close = np.any(d < 0.05, axis=1)

    # Gaps: d_{k+1} - d_k, shape (n, n-2)
    gaps = d[:, 1:] - d[:, :-1]
    k_idx = np.argmin(gaps, axis=1)  # (n,)

    # D = (d_k + d_{k+1}) / 2
    D = (d[np.arange(n), k_idx] + d[np.arange(n), k_idx + 1]) / 2.0

    # Validity mask
    valid = (~has_close) & (D >= 0.1)

    # Select 4 distances based on k
    n_minus_3 = n - 3  # 17

    selected = np.zeros((n, 4))

    mask_k0 = valid & (k_idx == 0)
    mask_k17 = valid & (k_idx == n_minus_3)
    mask_mid = valid & (k_idx > 0) & (k_idx < n_minus_3)

    # k = 0: d[0], d[1], d[2], d[3]
    for i in np.where(mask_k0)[0]:
        selected[i] = d[i, 0:4]

    # k = n-3 (17): d[15], d[16], d[17], d[18]
    for i in np.where(mask_k17)[0]:
        selected[i] = d[i, 15:19]

    # Middle: d[k-1], d[k], d[k+1], d[k+2]
    for i in np.where(mask_mid)[0]:
        k = k_idx[i]
        selected[i] = d[i, k - 1:k + 3]

    # raw = mean of max(d/D, D/d) over the 4 selected
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.maximum(selected / D[:, np.newaxis], D[:, np.newaxis] / selected)
    raw = np.mean(ratios, axis=1)

    # score = min(1, 1/raw), only for valid
    with np.errstate(divide='ignore', invalid='ignore'):
        score_vals = np.minimum(1.0, 1.0 / raw)
    scores = np.where(valid, score_vals, 0.0)

    return scores

def _min_vertex_score(radii, check_convexity=True):
    """
    radii: array of 20 radii (positive).
    Returns the minimum vertex score after normalization.
    If check_convexity is True, returns 0 if not strictly convex.
    """
    angles = _ANGLES
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack((x, y))

    # Normalize
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    points = points / mean_dist

    if check_convexity:
        # Convexity check
        p = points
        p_next = np.roll(p, -1, axis=0)
        p_next2 = np.roll(p, -2, axis=0)
        e1 = p_next - p
        e2 = p_next2 - p_next
        cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]

        if np.any(cross <= 0):  # Strictly convex: cross > 0 (assuming CCW)
            return 0.0
        
        # No repeated points
        if np.min(np.linalg.norm(p_next - p, axis=1)) < 1e-9:
            return 0.0

    scores = _compute_scores(points)
    return float(np.min(scores))

def _convexity_penalty(radii):
    """
    Returns a penalty if the polygon is not strictly convex.
    """
    angles = _ANGLES
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack((x, y))

    # Normalize
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 100.0
    points = points / mean_dist

    p = points
    p_next = np.roll(p, -1, axis=0)
    p_next2 = np.roll(p, -2, axis=0)
    e1 = p_next - p
    e2 = p_next2 - p_next
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    
    # Penalty for cross <= 0
    viol = np.maximum(0, -cross)
    return float(np.sum(viol ** 2))

def equidistant_points_polygon20() -> np.ndarray:
    _init()
    
    n = 20
    angles = _ANGLES
    
    # Baseline: Regular polygon
    reg_radii = np.ones(n)
    reg_score = _min_vertex_score(reg_radii)
    
    best_score = reg_score
    best_radii = reg_radii.copy()
    
    def objective(r):
        r = np.maximum(r, 0.05)
        score = _min_vertex_score(r, check_convexity=False)
        pen = _convexity_penalty(r)
        # If convex, pen is 0. If not, score is likely 0 or low, pen is high.
        # We want to maximize score, so minimize -score + lambda * pen
        return -score + 50.0 * pen

    # Initializations
    inits = []
    inits.append(np.ones(n))
    inits.append(np.maximum(1.0 + 0.15 * np.cos(2 * angles), 0.05))
    inits.append(np.maximum(1.0 + 0.15 * np.cos(4 * angles), 0.05))
    inits.append(np.maximum(1.0 + 0.15 * np.cos(5 * angles), 0.05))
    inits.append(np.maximum(1.0 + 0.10 * np.cos(3 * angles), 0.05))
    rng = np.random.default_rng(12345)
    inits.append(np.maximum(1.0 + 0.08 * rng.standard_normal(n), 0.05))
    rng2 = np.random.default_rng(67890)
    inits.append(np.maximum(1.0 + 0.12 * rng2.standard_normal(n), 0.05))
    inits.append(np.maximum(1.0 + 0.10 * np.cos(6 * angles), 0.05))
    inits.append(np.maximum(1.0 + 0.10 * np.cos(7 * angles), 0.05))
    inits.append(np.maximum(1.0 + 0.05 * np.arange(n) / n, 0.05))
    
    # Probe eval time
    probe_start = time.time()
    _min_vertex_score(np.ones(n))
    eval_time = max(time.time() - probe_start, 1e-4)
    
    max_iter_cap = 5000
    
    for idx, x0 in enumerate(inits):
        if time.time() > _DEADLINE - 10:
            break
            
        remaining = _DEADLINE - time.time()
        max_iter = min(max_iter_cap, max(200, int(remaining * 0.4 / max(eval_time * 20, 1e-3))))
        
        try:
            res = minimize(
                objective, x0, method='Nelder-Mead',
                options={
                    'maxiter': max_iter,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True,
                }
            )
            
            # Check the actual score of the result (without penalty)
            res_radii = np.maximum(res.x, 0.05)
            res_score = _min_vertex_score(res_radii)
            
            if res_score > best_score:
                best_score = res_score
                best_radii = res_radii.copy()
                
        except Exception:
            pass

    # Polish phase
    if time.time() < _DEADLINE - 30 and best_score > 0:
        remaining = _DEADLINE - time.time()
        max_iter = min(max_iter_cap, int(remaining * 0.7 / max(eval_time * 20, 1e-3)))
        try:
            res2 = minimize(
                objective, best_radii, method='Nelder-Mead',
                options={
                    'maxiter': max_iter,
                    'xatol': 1e-12,
                    'fatol': 1e-12,
                    'adaptive': True,
                }
            )
            res2_radii = np.maximum(res2.x, 0.05)
            res2_score = _min_vertex_score(res2_radii)
            if res2_score > best_score:
                best_score = res2_score
                best_radii = res2_radii.copy()
        except Exception:
            pass

    # Construct final polygon
    radii = best_radii
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    polygon = np.column_stack((x, y))

    # Final normalization
    centroid = np.mean(polygon, axis=0)
    polygon = polygon - centroid
    dists = np.linalg.norm(polygon, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist > 0.01:
        polygon = polygon / mean_dist

    if not np.all(np.isfinite(polygon)):
        polygon = np.column_stack((np.cos(angles), np.sin(angles)))

    return polygon
