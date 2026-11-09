import numpy as np
import time
from scipy.optimize import minimize

def _vertex_score(pts, i):
    """Compute the vertex score for vertex i given all points."""
    n = len(pts)
    d = np.linalg.norm(pts - pts[i], axis=1)
    d[i] = np.inf  # exclude self
    d_sorted = np.sort(d)  # d_0 <= ... <= d_{n-2}
    
    # Check if any distance < 0.05
    if d_sorted[0] < 0.05:
        return 0.0
    
    # Find k: first index minimizing d_{k+1} - d_k
    gaps = np.diff(d_sorted)  # gaps[j] = d_sorted[j+1] - d_sorted[j]
    k = int(np.argmin(gaps))
    D = (d_sorted[k] + d_sorted[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
    
    # Select four distances
    if k == 0:
        selected = d_sorted[0:4]
    elif k == n - 3:  # k == 17 for n=20
        selected = d_sorted[n-5:n-1]  # d_15..d_18
    else:
        selected = d_sorted[k-1:k+3]  # d_{k-1}..d_{k+2}
    
    # raw = mean of max(d/D, D/d)
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    
    return min(1.0, 1.0 / raw)

def _min_vertex_score(pts):
    """Compute the minimum vertex score over all vertices."""
    scores = [_vertex_score(pts, i) for i in range(len(pts))]
    return min(scores)

def _params_to_polygon(params, n=20):
    """Convert 40 params (angles + log_radii) to polygon vertices."""
    angles = params[:n]
    log_radii = params[n:]
    radii = np.exp(log_radii)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def _neg_min_score(params, n=20):
    """Negative of minimum vertex score for minimization."""
    pts = _params_to_polygon(params, n)
    # Normalize: subtract centroid, divide by mean distance to centroid
    centroid = np.mean(pts, axis=0)
    pts_norm = pts - centroid
    dists = np.linalg.norm(pts_norm, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return 1.0  # bad, return large value
    pts_norm = pts_norm / mean_dist
    score = _min_vertex_score(pts_norm)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, use 630s wall clock
    
    # Initialize with regular polygon
    angles0 = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radii0 = np.ones(n)
    params0 = np.concatenate([angles0, np.log(radii0)])
    
    # Initial polygon
    pts0 = _params_to_polygon(params0, n)
    centroid = np.mean(pts0, axis=0)
    pts0_norm = pts0 - centroid
    mean_dist = np.mean(np.linalg.norm(pts0_norm, axis=1))
    pts0_norm = pts0_norm / mean_dist
    best_score = _min_vertex_score(pts0_norm)
    best_params = params0.copy()
    
    # Try optimization with time guard
    try:
        # Use Nelder-Mead for derivative-free optimization
        result = minimize(
            _neg_min_score,
            params0,
            args=(n,),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        
        if result.fun < -best_score:
            best_params = result.x.copy()
            best_score = -result.fun
        
        # Check time
        if time.time() > deadline - 60:
            pass  # skip further optimization
        
        # Try a few more restarts if time permits
        n_restarts = 3
        for r in range(n_restarts):
            if time.time() > deadline - 60:
                break
            
            # Perturb from best
            np.random.seed(42 + r)
            perturb = np.random.normal(0, 0.05, size=n * 2)
            params_r = best_params + perturb
            
            result_r = minimize(
                _neg_min_score,
                params_r,
                args=(n,),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if result_r.fun < -best_score:
                best_params = result_r.x.copy()
                best_score = -result_r.fun
    
    except Exception:
        pass
    
    # Build final polygon from best params
    pts = _params_to_polygon(best_params, n)
    
    # Normalize
    centroid = np.mean(pts, axis=0)
    pts_norm = pts - centroid
    dists = np.linalg.norm(pts_norm, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        # Fallback to regular polygon
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        return np.stack([x, y], axis=1)
    pts_norm = pts_norm / mean_dist
    
    # Verify convexity and validity
    # Check no repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pts_norm[i] - pts_norm[j]) < 1e-6:
                angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
                x = np.cos(angles)
                y = np.sin(angles)
                return np.stack([x, y], axis=1)
    
    # Check convexity: all cross products have same sign
    cross_products = []
    for i in range(n):
        p0 = pts_norm[i]
        p1 = pts_norm[(i+1) % n]
        p2 = pts_norm[(i+2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        cross_products.append(cross)
    
    cross_products = np.array(cross_products)
    if np.any(np.abs(cross_products) < 1e-10):
        # Not strictly convex, fallback
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        return np.stack([x, y], axis=1)
    
    if not (np.all(cross_products > 0) or np.all(cross_products < 0)):
        # Not convex, fallback
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        return np.stack([x, y], axis=1)
    
    return pts_norm
