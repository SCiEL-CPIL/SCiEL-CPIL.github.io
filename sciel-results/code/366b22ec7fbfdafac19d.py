import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_N = 20

def _build_points(params):
    """
    params: (2*N,) array. First N are radii, next N are angles.
    """
    radii = params[:_N]
    angles = params[_N:]
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

def _normalize(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return None
    return points / mean_dist

def _is_strictly_convex(points):
    # Check cross products
    p1 = np.roll(points, -1, axis=0)
    p2 = np.roll(points, -2, axis=0)
    e1 = p1 - points
    e2 = p2 - p1
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    if np.any(np.abs(cross) < 1e-12):
        return False
    if np.any(cross > 0) and np.any(cross < 0):
        return False
    # Check for repeated points
    for i in range(_N):
        for j in range(i+1, _N):
            if np.linalg.norm(points[i] - points[j]) < 1e-9:
                return False
    return True

def _compute_scores(points):
    n = len(points)
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    
    scores = np.zeros(n)
    np.fill_diagonal(dists, np.inf)
    
    for i in range(n):
        d = np.sort(dists[i, :])
        d = d[:n-1]  # Exclude self
        
        if np.any(d < 0.05):
            scores[i] = 0.0
            continue
            
        gaps = np.diff(d)
        k = np.argmin(gaps)
        D = (d[k] + d[k+1]) / 2.0
        
        if D < 0.1:
            scores[i] = 0.0
            continue
            
        if k == 0:
            sel = d[0:4]
        elif k == n - 3:  # 17 for N=20
            sel = d[15:19]
        else:
            sel = d[k-1:k+3]
            
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.maximum(sel / D, D / sel)
        raw = np.mean(ratios)
        scores[i] = np.min(1.0, 1.0 / raw)
        
    return scores

def _min_score(params):
    points = _build_points(params)
    norm_pts = _normalize(points)
    if norm_pts is None:
        return -1000.0
    if not _is_strictly_convex(norm_pts):
        return -1000.0
    scores = _compute_scores(norm_pts)
    return np.min(scores)

def _regular_polygon_params():
    """Return parameters for a regular 20-gon."""
    angles = np.linspace(0, 2*np.pi, _N, endpoint=False)
    radii = np.ones(_N)
    return np.concatenate([radii, angles])

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s limit
    
    # Start from regular polygon
    best_params = _regular_polygon_params()
    best_score = _min_score(best_params)
    
    # Bounds for DE
    # Radii: 0.5 to 1.5 (relative to regular polygon)
    # Angles: regular angle ± 0.5 radians
    regular_angles = np.linspace(0, 2*np.pi, _N, endpoint=False)
    bounds = [(0.5, 1.5)] * _N
    for i in range(_N):
        bounds.append((regular_angles[i] - 0.5, regular_angles[i] + 0.5))
    
    # Multi-start DE with different seeds and perturbation levels
    seeds = [42, 123, 456, 789, 101]
    perturbation_scales = [0.1, 0.3, 0.5, 0.8, 1.0]
    
    for seed, scale in zip(seeds, perturbation_scales):
        if time.time() > deadline - 60:
            break
        
        # Create initial population with perturbations around regular polygon
        def objective(params):
            s = _min_score(params)
            if s < 0:
                return 1000.0 - s
            return -s
        
        # Run DE with smaller population for speed
        try:
            res = differential_evolution(
                objective,
                bounds,
                maxiter=15,
                popsize=25,
                seed=seed,
                tol=1e-4,
                atol=1e-4,
                init='sobol'
            )
            cand_params = res.x
            cand_score = _min_score(cand_params)
            
            if cand_score > best_score:
                best_score = cand_score
                best_params = cand_params.copy()
                
        except Exception:
            continue
        
        # Refine with Nelder-Mead if time allows
        if time.time() > deadline - 30:
            break
        
        try:
            res_nm = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            refine_params = res_nm.x
            refine_score = _min_score(refine_params)
            
            if refine_score > best_score:
                best_score = refine_score
                best_params = refine_params.copy()
                
        except Exception:
            continue
    
    # Final refinement with tight tolerances if time allows
    if time.time() < deadline - 15:
        def objective_final(params):
            s = _min_score(params)
            if s < 0:
                return 1000.0 - s
            return -s
        
        try:
            res_final = minimize(
                objective_final,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            final_params = res_final.x
            final_score = _min_score(final_params)
            
            if final_score > best_score:
                best_score = final_score
                best_params = final_params.copy()
                
        except Exception:
            pass
    
    # Build and return final points
    final_points = _build_points(best_params)
    norm_final = _normalize(final_points)
    
    if norm_final is None or not _is_strictly_convex(norm_final):
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, _N, endpoint=False)
        radii = np.ones(_N)
        norm_final = _normalize(np.column_stack([radii*np.cos(angles), radii*np.sin(angles)]))
        
    return norm_final
