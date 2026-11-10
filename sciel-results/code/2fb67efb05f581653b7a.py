import numpy as np
import time
from scipy.optimize import minimize

def compute_scores(polygon):
    n = 20
    scores = np.zeros(n)
    for i in range(n):
        v = polygon[i]
        dists = np.linalg.norm(polygon - v, axis=1)
        # Exclude self
        dists[i] = np.inf
        # Sort
        dists = np.sort(dists)
        # d_0 ... d_18
        if dists[0] < 0.05:
            scores[i] = 0.0
            continue
        
        # Find k: first index minimizing d_{k+1} - d_k
        diffs = np.diff(dists)
        k = np.argmin(diffs)
        
        D = (dists[k] + dists[k+1]) / 2.0
        if D < 0.1:
            scores[i] = 0.0
            continue
            
        # Select 4 distances
        if k == 0:
            d_sel = dists[0:4]
        elif k == 17: # n-2 = 18, so max k is 18? 
            # dists has 19 elements. indices 0..18.
            # diffs has 18 elements. indices 0..17.
            # k is in 0..17.
            # if k=17, take d_15..d_18
            d_sel = dists[15:19]
        else:
            d_sel = dists[k-1:k+3] # k-1, k, k+1, k+2
            
        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(d_sel / D, D / d_sel)
        raw = np.mean(ratios)
        scores[i] = min(1.0, 1.0 / raw)
        
    return scores

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
    n = len(pts)
    # Check finite
    if not np.all(np.isfinite(pts)):
        return False
    # Check no repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-6:
                return False
    # Check convexity
    # Cross products of consecutive edges
    v_curr = pts
    v_next = np.roll(pts, -1, axis=0)
    v_next2 = np.roll(pts, -2, axis=0)
    e1 = v_next - v_curr
    e2 = v_next2 - v_next
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    # All must be same sign and nonzero
    if np.any(cross <= 1e-6):
        return False
    # Check total turning is 2*pi (implicit in convexity for simple polygon)
    return True

def _normalize(pts):
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    mean_dist = np.mean(np.linalg.norm(pts, axis=1))
    if mean_dist <= 0.01:
        return None
    return pts / mean_dist

def _param_to_pts(params):
    n = 20
    angles = params[:n]
    radii = params[n:]
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

def _objective(params):
    pts = _param_to_pts(params)
    pts_norm = _normalize(pts)
    if pts_norm is None:
        return 1000.0
    if not _is_valid(pts_norm):
        # Penalty for non-convexity
        # Calculate how much it violates convexity
        v_curr = pts_norm
        v_next = np.roll(pts_norm, -1, axis=0)
        v_next2 = np.roll(pts_norm, -2, axis=0)
        e1 = v_next - v_curr
        e2 = v_next2 - v_next
        cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        penalty = np.sum(np.minimum(cross, 0)**2)
        return 100.0 + penalty
    score = _min_vertex_score(pts_norm)
    return -score
    # polygon: (20, 2)
    n = 20
    scores = np.zeros(n)
    for i in range(n):
        v = polygon[i]
        dists = np.linalg.norm(polygon - v, axis=1)
        # Exclude self
        dists[i] = np.inf
        # Sort
        dists = np.sort(dists)
        # d_0 ... d_18
        if dists[0] < 0.05:
            scores[i] = 0.0
            continue
        
        # Find k: first index minimizing d_{k+1} - d_k
        diffs = np.diff(dists)
        k = np.argmin(diffs)
        
        D = (dists[k] + dists[k+1]) / 2.0
        if D < 0.1:
            scores[i] = 0.0
            continue
            
        # Select 4 distances
        if k == 0:
            d_sel = dists[0:4]
        elif k == 17: # n-2 = 18, so max k is 18? 
            # dists has 19 elements. indices 0..18.
            # diffs has 18 elements. indices 0..17.
            # k is in 0..17.
            # if k=17, take d_15..d_18
            d_sel = dists[15:19]
        else:
            d_sel = dists[k-1:k+3] # k-1, k, k+1, k+2
            
        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(d_sel / D, D / d_sel)
        raw = np.mean(ratios)
        scores[i] = min(1.0, 1.0 / raw)
        
    return scores

def objective(params):
    return _objective(params)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = 630.0 - 10.0 # Safety margin
    
    n = 20
    # Initial guess: Regular polygon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radii = np.ones(n)
    x0 = np.concatenate([angles, radii])
    
    best_params = x0
    best_score = -1e9
    
    # Run Nelder-Mead
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        # Verify the result
        angles_opt = res.x[:20]
        radii_opt = res.x[20:]
        
        x = radii_opt * np.cos(angles_opt)
        y = radii_opt * np.sin(angles_opt)
        polygon = np.column_stack((x, y))
        
        # Normalize
        polygon_norm = _normalize(polygon)
        if polygon_norm is None:
             return np.column_stack((np.cos(angles), np.sin(angles))) # Fallback to regular
        
        if not _is_valid(polygon_norm):
             # Fallback to regular
             return np.column_stack((np.cos(angles), np.sin(angles)))
             
        return polygon_norm
    except Exception:
        # Fallback to regular
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
