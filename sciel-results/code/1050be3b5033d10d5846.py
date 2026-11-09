# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _vertex_score(dists, k):
    """Compute vertex score given sorted distances and gap index k."""
    d = np.sort(dists)
    if d[0] < 0.05:
        return 0.0
    gaps = d[1:] - d[:-1]
    k = int(np.argmin(gaps))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        four = d[0:4]
    elif k == 17:
        four = d[15:19]
    else:
        four = d[k-1:k+3]
    raw = np.mean(np.maximum(four / D, D / four))
    return min(1.0, 1.0 / raw)

def _compute_min_score(polygon):
    """Compute the minimum vertex score for a given polygon."""
    n = len(polygon)
    # Normalize
    centroid = polygon.mean(axis=0)
    p = polygon - centroid
    dists_to_centroid = np.linalg.norm(p, axis=1)
    mean_dist = dists_to_centroid.mean()
    if mean_dist <= 0.01:
        return 0.0
    p = p / mean_dist
    
    # Check convexity: all turns same sign and nonzero
    angles = np.arctan2(p[:,1], p[:,0])
    # Sort by angle to ensure cyclic order (should already be sorted)
    order = np.argsort(angles)
    p_sorted = p[order]
    
    # Check turns
    min_score = 1.0
    for i in range(n):
        d = np.linalg.norm(p_sorted[i] - p_sorted, axis=1)
        d[i] = np.inf  # exclude self
        score = _vertex_score(d, None)
        if score < min_score:
            min_score = score
        if min_score <= 0.0:
            break
    return min_score

def _polygon_from_angles(angles, radii=None):
    """Create polygon from angles (and optional radii)."""
    n = len(angles)
    if radii is None:
        radii = np.ones(n)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def _is_convex(polygon):
    """Check if polygon is strictly convex (all cross products same sign)."""
    n = len(polygon)
    if n < 3:
        return False
    signs = []
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i+1) % n]
        p2 = polygon[(i+2) % n]
        cross = (p1[0]-p0[0])*(p2[1]-p1[1]) - (p1[1]-p0[1])*(p2[0]-p1[0])
        if abs(cross) < 1e-12:
            return False
        signs.append(np.sign(cross))
    return len(set(signs)) == 1

def _objective(params):
    """Objective: negative of minimum vertex score. params = [angles..., radii...]."""
    n = 20
    angles = params[:n]
    radii = params[n:]
    
    # Ensure angles are sorted (for convexity)
    angles_sorted = np.sort(angles)
    # Sort radii correspondingly
    order = np.argsort(angles)
    radii_sorted = radii[order]
    
    polygon = _polygon_from_angles(angles_sorted, radii_sorted)
    
    if not _is_convex(polygon):
        return 1.0  # penalty for non-convex
    
    score = _compute_min_score(polygon)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s limit
    
    # Best result so far: regular polygon
    angles_reg = np.linspace(0.0, 2.0*np.pi, n, endpoint=False)
    best_polygon = _polygon_from_angles(angles_reg)
    best_score = _compute_min_score(best_polygon)
    
    # Multiple restarts with L-BFGS-B
    np.random.seed(42)
    n_restarts = 20
    max_iter = 500
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate diverse initial points
        if restart == 0:
            # Start from regular
            angles0 = angles_reg.copy()
            radii0 = np.ones(n)
        elif restart == 1:
            # Slightly perturbed regular
            angles0 = angles_reg + np.random.normal(0, 0.05, n)
            radii0 = 1.0 + np.random.normal(0, 0.1, n)
        elif restart == 2:
            # More perturbed
            angles0 = angles_reg + np.random.normal(0, 0.15, n)
            radii0 = 1.0 + np.random.normal(0, 0.3, n)
        else:
            # Random convex-ish start
            angles0 = np.sort(np.random.uniform(0, 2*np.pi, n))
            # Ensure minimum angular separation
            for i in range(n-1):
                if angles0[i+1] - angles0[i] < 0.1:
                    angles0[i+1] = angles0[i] + 0.1
            if angles0[0] + 2*np.pi - angles0[-1] < 0.1:
                angles0[0] += 0.1
            radii0 = 1.0 + np.random.normal(0, 0.5, n)
            radii0 = np.clip(radii0, 0.5, 2.0)
        
        params0 = np.concatenate([angles0, radii0])
        
        # Bounds
        angle_bounds = [(0, 2*np.pi)] * n
        radius_bounds = [(0.3, 3.0)] * n
        bounds = angle_bounds + radius_bounds
        
        try:
            result = minimize(
                _objective,
                params0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception:
            continue
        
        if time.time() > deadline:
            break
        
        if result.success or result.fun < _objective(params0):
            params_best = result.x
            angles_opt = np.sort(params_best[:n])
            order = np.argsort(params_best[:n])
            radii_opt = params_best[n:][order]
            
            polygon_opt = _polygon_from_angles(angles_opt, radii_opt)
            
            if _is_convex(polygon_opt):
                score_opt = _compute_min_score(polygon_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_polygon = polygon_opt
    
    return best_polygon

# EVOLVE-BLOCK-END
