# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _vertex_score(distances):
    """
    distances: array of 19 distances from vertex i to others.
    Returns score for this vertex in [0, 1].
    """
    d = np.sort(distances)
    
    if np.any(d < 0.05):
        return 0.0
        
    diffs = d[1:] - d[:-1]
    k = np.argmin(diffs)
    
    D = (d[k] + d[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
        
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k-1:k+3]
        
    if len(selected) != 4:
        return 0.0

    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    
    score = min(1.0, 1.0 / raw)
    return score

def _is_valid_convex(points):
    """Check if points form a strictly convex polygon."""
    n = len(points)
    
    # Check for repeated points
    # Vectorized check for min distance between any pair
    # This is O(n^2) but n=20 is small
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-6:
        return False
        
    # Check turns
    edges = np.roll(points, -1, axis=0) - points
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    
    if np.any(np.abs(cross) < 1e-8):
        return False
        
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return False
        
    return True

def _polygon_score(points):
    """
    points: (20, 2) array.
    Returns minimum vertex score after normalization.
    """
    centroid = np.mean(points, axis=0)
    points_norm = points - centroid
    
    dists_to_c = np.linalg.norm(points_norm, axis=1)
    mean_dist = np.mean(dists_to_c)
    
    if mean_dist <= 0.01:
        return 0.0
        
    points_norm = points_norm / mean_dist
    
    if not _is_valid_convex(points_norm):
        return 0.0
        
    n = len(points_norm)
    min_score = 1.0
    for i in range(n):
        other_indices = [j for j in range(n) if j != i]
        dists = np.linalg.norm(points_norm[other_indices] - points_norm[i], axis=1)
        s = _vertex_score(dists)
        if s < min_score:
            min_score = s
        if min_score == 0.0:
            return 0.0
            
    return min_score

def _regular_polygon_points():
    n = 20
    radius = 1.0
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.stack([x, y], axis=1)

def _params_to_points(params):
    """
    params: array of length 40 - first 20 are radii, next 20 are angular offsets (in radians)
    Returns (20, 2) array of points.
    """
    n = 20
    radii = params[:n]
    angle_offsets = params[n:]
    
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    angles = base_angles + angle_offsets
    
    # Ensure angles are sorted and wrapped to maintain convexity order
    # We sort the angles to ensure the polygon vertices are in cyclic order
    # This makes the mapping from params to points non-smooth at boundaries,
    # but Nelder-Mead can handle it.
    angles = np.sort(angles)
    angles = (angles % (2.0 * np.pi))
    angles = np.sort(angles)
    
    # Ensure radii are positive to avoid points crossing the origin
    radii = np.maximum(radii, 0.1)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def _objective(params):
    points = _params_to_points(params)
    score = _polygon_score(points)
    # Return negative score for minimization
    # If score is 0 (invalid), return a large penalty to guide search away
    if score == 0.0:
        return 100.0
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that tries to be a counterexample to the
    conjecture "every convex polygon has a vertex v with no 4 other vertices equidistant
    from v". The goal is a polygon where EVERY vertex has (at least) 4 other vertices that
    are approximately equidistant from it. Quality is the minimum, over all vertices, of a
    per-vertex equidistance score in [0, 1] (1.0 = perfect counterexample).

    This baseline is the upstream search's starting construction: a regular polygon whose
    vertices lie evenly on a circle (deterministic, strictly convex). The scoring function
    is scale/translation invariant, so the circle radius is arbitrary.

    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    deadline = 620.0  # Safety margin before 630s limit
    
    n = 20
    
    # Baseline: regular polygon
    best_points = _regular_polygon_points()
    best_score = _polygon_score(best_points)
    
    # Initialize best params to correspond to regular polygon
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    best_params = np.concatenate([np.ones(n), np.zeros(n)])
    
    # Basin Hopping Strategy
    # We will perform a series of "hops". Each hop involves:
    # 1. Perturbing the current best parameters.
    # 2. Running a local optimization (Nelder-Mead) from the perturbed state.
    # 3. Accepting the result if it improves the score.
    
    max_hops = 500
    hop_noise_scale = 0.1
    
    for hop in range(max_hops):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb the best parameters
        # Use a mix of small and large perturbations
        scale = hop_noise_scale * (1.0 + 0.5 * np.sin(hop)) # Vary scale slightly
        r_pert = scale * np.random.randn(n)
        a_pert = scale * 0.5 * np.random.randn(n) # Angles are more sensitive
        current_params = best_params + np.concatenate([r_pert, a_pert])
        
        # Local Optimization
        try:
            res = minimize(
                _objective, 
                current_params, 
                method='Nelder-Mead', 
                options={'xatol': 1e-5, 'fatol': 1e-6, 'maxiter': 200, 'maxfev': 1000}
            )
        except Exception:
            continue
        
        r_opt = res.x
        points = _params_to_points(r_opt)
        score = _polygon_score(points)
        
        if score > best_score:
            best_score = score
            best_points = points
            best_params = r_opt
            # Reset hop noise scale if we found a significant improvement?
            # Or just keep going.
            
    return best_points

# EVOLVE-BLOCK-END
