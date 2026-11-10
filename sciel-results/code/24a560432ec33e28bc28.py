import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _vertex_score(v, pts):
    """Compute the vertex score for vertex v given all points pts."""
    diffs = pts - v
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    mask = np.ones(len(pts), dtype=bool)
    mask[np.argmin(dists)] = False
    d = np.sort(dists[mask])
    if np.any(d < 0.05):
        return 0.0
    gaps = np.diff(d)
    k = int(np.argmin(gaps))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k - 1:k + 3]
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def _min_vertex_score(pts):
    scores = [_vertex_score(pts[i], pts) for i in range(len(pts))]
    return min(scores)

def _is_valid(pts):
    if len(pts) != 20:
        return False
    if not np.all(np.isfinite(pts)):
        return False
    for i in range(20):
        for j in range(i + 1, 20):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-12:
                return False
    signs = []
    for i in range(20):
        p0 = pts[i]
        p1 = pts[(i + 1) % 20]
        p2 = pts[(i + 2) % 20]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-14:
            return False
        signs.append(np.sign(cross))
    if not all(s == signs[0] for s in signs):
        return False
    return True

def _normalize(pts):
    centroid = np.mean(pts, axis=0)
    pts_norm = pts - centroid
    dists = np.linalg.norm(pts_norm, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return pts_norm, False
    pts_norm = pts_norm / mean_dist
    return pts_norm, True

def _params_to_pts(params):
    """
    Convert radii to polygon vertices.
    params[0]: Base radius R
    params[1:21]: Individual radii r_i for each vertex
    """
    n = 20
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    R = params[0]
    r = R * params[1:21]
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])

def _soft_min(scores, beta=100.0):
    """
    Smooth approximation of min(scores) using log-sum-exp.
    min(s) ≈ -1/beta * log(sum(exp(-beta * s)))
    """
    # Clamp to avoid overflow/underflow
    scores = np.clip(scores, 1e-10, 1.0)
    neg_beta_scores = -beta * scores
    max_val = np.max(neg_beta_scores)
    sum_exp = np.sum(np.exp(neg_beta_scores - max_val))
    log_sum_exp = max_val + np.log(sum_exp)
    return -log_sum_exp / beta

def _smooth_objective(params, deadline):
    if time.time() > deadline:
        return 1e10
    
    pts = _params_to_pts(params)
    norm_pts, valid = _normalize(pts)
    if not valid:
        return 1e10
    
    if not _is_valid(norm_pts):
        # Penalty for invalidity
        # Calculate a smooth penalty based on cross product violations
        penalty = 0.0
        for i in range(20):
            p0 = norm_pts[i]
            p1 = norm_pts[(i + 1) % 20]
            p2 = norm_pts[(i + 2) % 20]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if cross < 0:
                penalty += cross * cross
        return 1e5 + penalty
    
    scores = np.array([_vertex_score(norm_pts[i], norm_pts) for i in range(20)])
    # We want to maximize min score, so minimize -soft_min
    return -_soft_min(scores)

def _exact_objective(params):
    pts = _params_to_pts(params)
    norm_pts, valid = _normalize(pts)
    if not valid:
        return 1e10
    if not _is_valid(norm_pts):
        return 1e10
    score = _min_vertex_score(norm_pts)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    # Initial best: Regular 20-gon
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    norm_reg, _ = _normalize(pts_reg)
    best_score = _min_vertex_score(norm_reg)
    best_pts = pts_reg.copy()
    
    # Phase 1: Differential Evolution with smooth surrogate
    # Bounds: R in [0.5, 1.5], radii in [0.5, 1.5]
    bounds = [(0.5, 1.5)] + [(-0.2, 0.2)] * 20
    # Note: params[1:21] are deviations from 1.0? Or absolute radii?
    # Let's use absolute radii for simplicity in DE, but constrain them to be positive.
    # Actually, _params_to_pts uses R * r_i. If R=1, r_i=1 is regular.
    # Let's optimize r_i directly in [0.5, 1.5] and R=1 fixed? Or optimize R and r_i?
    # The prompt says "polar coordinate parameterization (angles+radii)".
    # Let's stick to R and 20 radii.
    bounds_de = [(0.8, 1.2)] + [(0.8, 1.2)] * 20
    
    try:
        res = differential_evolution(
            _smooth_objective,
            bounds_de,
            args=(deadline,),
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            init='latinhypercube',
            polish=False # We will do our own polish
        )
        
        if res.fun < 1e5: # If it found something valid
            trial_pts = _params_to_pts(res.x)
            norm_trial, valid = _normalize(trial_pts)
            if valid and _is_valid(norm_trial):
                trial_score = _min_vertex_score(norm_trial)
                if trial_score > best_score:
                    best_score = trial_score
                    best_pts = trial_pts.copy()
                    best_params = res.x.copy()
                else:
                    best_params = res.x.copy()
            else:
                best_params = res.x.copy()
        else:
            # Fallback to regular
            best_params = np.concatenate([[1.0], np.ones(20)])
            
    except Exception:
        best_params = np.concatenate([[1.0], np.ones(20)])
    
    # Phase 2: Nelder-Mead Polish on exact objective
    if time.time() < deadline - 10:
        try:
            # Start from best_params
            res_nm = minimize(
                _exact_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if res_nm.fun < 1e5:
                trial_pts = _params_to_pts(res_nm.x)
                norm_trial, valid = _normalize(trial_pts)
                if valid and _is_valid(norm_trial):
                    trial_score = _min_vertex_score(norm_trial)
                    if trial_score > best_score:
                        best_score = trial_score
                        best_pts = trial_pts.copy()
        except Exception:
            pass
            
    # Phase 3: Final check and return
    norm_best, valid = _normalize(best_pts)
    if valid and _is_valid(norm_best):
        final_score = _min_vertex_score(norm_best)
        if final_score > best_score:
            best_score = final_score
            best_pts = norm_best.copy() # Return normalized? No, return points.
            # The task says "returning a (20, 2) array of finite vertex coordinates".
            # It doesn't say normalized, but the metric normalizes it.
            # Returning the normalized version is safe and ensures consistency.
            # However, the original program returned best_pts which were raw.
            # Let's return the raw points that produced the best score.
            # Actually, _normalize is part of the scoring. The returned points are scored by normalizing them.
            # So returning best_pts (raw) is correct.
            
    return best_pts
