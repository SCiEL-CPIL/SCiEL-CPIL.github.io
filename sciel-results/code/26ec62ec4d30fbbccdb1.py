import numpy as np
import scipy.optimize
import time

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

def _is_convex(pts):
    """Check if pts form a strictly convex polygon."""
    n = len(pts)
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        p2 = pts[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross <= 1e-12: # Strictly convex, so cross must be > 0 (assuming CCW)
             # Note: orientation might be CW. We should check sign consistency.
             # But if we start with CCW regular polygon, it should stay CCW.
             # To be safe, let's check if all crosses have the same sign and are non-zero.
             pass
    # Better check:
    crosses = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        p2 = pts[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        crosses.append(v1[0] * v2[1] - v1[1] * v2[0])
    crosses = np.array(crosses)
    if np.any(crosses == 0):
        return False
    if not np.all(crosses > 0) and not np.all(crosses < 0):
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

def _score_from_radii(r):
    """Calculate score for a polygon defined by radii r at fixed angles."""
    angles = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    
    pts_norm, valid_norm = _normalize(pts)
    if not valid_norm:
        return 0.0
        
    if not _is_convex(pts_norm):
        return 0.0
        
    return _min_vertex_score(pts_norm)

def _objective(r):
    """Objective function for minimization."""
    score = _score_from_radii(r)
    # We want to maximize score, so minimize -score.
    # If score is 0 (invalid), we return a large positive number to discourage it.
    if score <= 0.0:
        return 1000.0
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Initial guess: regular polygon
    r0 = np.ones(20)
    
    best_r = r0.copy()
    best_score = _score_from_radii(best_r)
    
    # Try Nelder-Mead optimization
    # We might need to try multiple starts or just one with good options
    try:
        result = scipy.optimize.minimize(_objective, r0, method='Nelder-Mead', 
                                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if result.fun < -best_score: # result.fun is -score, so lower is better
            best_r = result.x
            best_score = -result.fun
            
        # Check time
        if time.time() > deadline - 60.0:
            pass # Skip further restarts
            
        # Try a few random restarts if time permits
        np.random.seed(42)
        for i in range(10):
            if time.time() > deadline - 60.0:
                break
                
            # Perturb the best solution found so far
            r_start = best_r + np.random.normal(0, 0.1, 20)
            # Ensure radii are positive
            r_start = np.abs(r_start) + 0.1
            
            res = scipy.optimize.minimize(_objective, r_start, method='Nelder-Mead', 
                                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < -best_score:
                best_r = res.x
                best_score = -res.fun
                
    except Exception:
        pass

    # Construct final points from best_r
    angles = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    pts = np.column_stack([best_r * np.cos(angles), best_r * np.sin(angles)])
    pts_norm, _ = _normalize(pts)
    
    return pts_norm
