# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _vertex_score(v, points):
    """Compute the score for a single vertex v given all points."""
    n = len(points)
    # Compute distances from v to all other vertices
    diffs = points - v
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    # Exclude self
    idx = np.arange(n)
    mask = np.ones(n, dtype=bool)
    # Find which index is v
    # Actually, v should be one of the points; let's just exclude the zero distance
    # But due to floating point, let's sort and take the 19 non-zero
    # Better: compute distances to all, then exclude the one that's ~0
    # Actually, let's just use all 20 distances and the smallest will be ~0
    # Wait, the task says "19 distances from v to the other vertices"
    # So we need to exclude v itself
    
    # Find index of v in points
    # v is points[i] for some i
    # Let's just compute all pairwise distances and take the row for v
    # But we don't have the index. Let's restructure.
    
    # Actually, let's pass the index
    pass

def _vertex_score_idx(i, points):
    """Compute the score for vertex i given all points."""
    n = len(points)
    diffs = points - points[i]
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    dists[i] = np.inf  # Exclude self
    dists = np.sort(dists)
    
    if np.any(dists < 0.05):
        return 0.0
    
    # Find k: first index minimizing d_{k+1} - d_k
    gaps = np.diff(dists)
    k = np.argmin(gaps)
    
    D = (dists[k] + dists[k+1]) / 2.0
    if D < 0.1:
        return 0.0
    
    if k == 0:
        four_d = dists[0:4]
    elif k == 17:
        four_d = dists[15:19]
    else:
        four_d = dists[k-1:k+3]
    
    # raw = mean over these four d of max(d/D, D/d)
    ratios = np.maximum(four_d / D, D / four_d)
    raw = np.mean(ratios)
    
    return min(1.0, 1.0 / raw)

def _is_valid_convex(points):
    """Check if points form a strictly convex polygon."""
    n = len(points)
    if n < 3:
        return False
    
    # Check no repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.allclose(points[i], points[j], atol=1e-10):
                return False
    
    # Check convexity: all cross products of consecutive edges have same sign
    signs = []
    for i in range(n):
        p0 = points[i]
        p1 = points[(i+1) % n]
        p2 = points[(i+2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross = e1[0]*e2[1] - e1[1]*e2[0]
        if abs(cross) < 1e-15:
            return False
        signs.append(np.sign(cross))
    
    # All signs must be the same
    if not all(s == signs[0] for s in signs):
        return False
    
    # Check total turning is +/- 2*pi
    # For a convex polygon, this should be automatic if all turns are same sign
    # But let's verify the angles sum
    total_turn = 0.0
    for i in range(n):
        p0 = points[i-1]
        p1 = points[i]
        p2 = points[(i+1) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        a1 = np.arctan2(e1[1], e1[0])
        a2 = np.arctan2(e2[1], e2[0])
        turn = a2 - a1
        # Normalize to [-pi, pi]
        while turn > np.pi:
            turn -= 2*np.pi
        while turn < -np.pi:
            turn += 2*np.pi
        total_turn += turn
    
    if abs(abs(total_turn) - 2*np.pi) > 1e-6:
        return False
    
    return True

def _normalize(points):
    """Normalize: subtract centroid, divide by mean distance to centroid."""
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists = np.sqrt(np.sum(points**2, axis=1))
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return points, 0.0
    points = points / mean_dist
    return points, mean_dist

def _polygon_score(points):
    """Compute the overall polygon score (minimum vertex score)."""
    points, _ = _normalize(points)
    if not _is_valid_convex(points):
        return 0.0
    
    n = len(points)
    min_score = 1.0
    for i in range(n):
        s = _vertex_score_idx(i, points)
        if s < min_score:
            min_score = s
    return min_score

def _params_to_points(params):
    """Convert parameters to polygon points.
    params: 20 angles (relative perturbations) and 20 radii
    We use angles = base_angles + perturbations, and radii as given.
    """
    n = 20
    base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    angle_perts = params[:n]
    radii = params[n:2*n]
    angles = base_angles + angle_perts
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def _objective(params):
    """Objective function to minimize (negative of polygon score)."""
    points = _params_to_points(params)
    score = _polygon_score(points)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that maximizes the minimum
    vertex equidistance score.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s safety margin
    
    n = 20
    best_score = 0.0
    best_params = None
    
    # Initial: regular polygon
    base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    init_params = np.zeros(2*n)
    init_params[n:] = 1.0  # radii = 1
    
    # Try multiple restarts with different initial conditions
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            params = init_params.copy()
        else:
            # Random perturbation
            params = init_params.copy()
            params[:n] = np.random.uniform(-0.1, 0.1, n)
            params[n:] = 1.0 + np.random.uniform(-0.1, 0.1, n)
        
        # Use Nelder-Mead for optimization (no gradient needed)
        try:
            result = minimize(
                _objective,
                params,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if result.fun < 0:  # score > 0
                s = -result.fun
                if s > best_score:
                    best_score = s
                    best_params = result.x.copy()
        except Exception as e:
            pass
        
        if time.time() > deadline:
            break
    
    # If no good result found, fall back to regular polygon
    if best_params is None:
        points = _params_to_points(init_params)
    else:
        points = _params_to_points(best_params)
    
    # Final normalization
    points, _ = _normalize(points)
    
    return points

# EVOLVE-BLOCK-END
