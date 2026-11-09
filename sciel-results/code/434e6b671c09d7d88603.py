# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.optimize import minimize_scalar

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

def _spiral_to_points(params):
    """
    params: array of length 3.
    params[0]: log-spiral growth rate (log(r_{i+1}/r_i))
    params[1]: base angle offset
    params[2]: initial radius
    
    Creates a logarithmic spiral: r_k = r0 * exp(k * growth), theta_k = theta0 + k * dtheta
    where dtheta = 2*pi/20 (equal angular spacing for convexity)
    """
    n = 20
    growth = params[0]
    theta0 = params[1]
    r0 = params[2]
    
    k = np.arange(n)
    radii = r0 * np.exp(k * growth)
    angles = theta0 + k * (2.0 * np.pi / n)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def _general_to_points(params):
    """
    params: array of length 40.
    First 20: log-radii (u_i). r_i = exp(u_i).
    Next 20: log-angular-gaps (v_i). 
    The gaps are exp(v_i), normalized to sum to 2*pi.
    This ensures angles are sorted and cover 2*pi.
    """
    n = 20
    log_radii = params[:n]
    log_gaps = params[n:]
    
    radii = np.exp(log_radii)
    
    # Ensure radii are not too small to avoid degenerate polygons
    radii = np.maximum(radii, 0.1)
    
    gaps = np.exp(log_gaps)
    # Normalize gaps to sum to 2*pi
    total_gap = np.sum(gaps)
    if total_gap == 0:
        gaps = np.ones(n)
    else:
        gaps = gaps / total_gap * (2.0 * np.pi)
        
    angles = np.cumsum(gaps)
    angles = angles - angles[0] # Start at 0
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def _objective_spiral(params):
    points = _spiral_to_points(params)
    score = _polygon_score(points)
    # Return negative score for minimization
    if score == 0.0:
        return 100.0
    return -score

def _objective_general(params):
    points = _general_to_points(params)
    score = _polygon_score(points)
    # Return negative score for minimization
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
    
    # Phase 1: Search in spiral parameter space (3 parameters)
    # This is much faster and may find good solutions
    np.random.seed(42)
    
    # Spiral bounds: growth in [-0.5, 0.5], theta0 in [0, 2*pi], r0 in [0.5, 2.0]
    spiral_bounds = [(-0.5, 0.5), (0.0, 2.0 * np.pi), (0.5, 2.0)]
    
    # Try multiple random starts in spiral space
    for i in range(50):
        if time.time() > deadline - 30.0:
            break
        
        init_spiral = np.array([
            np.random.uniform(*spiral_bounds[0]),
            np.random.uniform(*spiral_bounds[1]),
            np.random.uniform(*spiral_bounds[2])
        ])
        
        try:
            res = minimize(
                _objective_spiral,
                init_spiral,
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-7, 'maxiter': 200, 'maxfev': 1000}
            )
        except Exception:
            continue
        
        points = _spiral_to_points(res.x)
        score = _polygon_score(points)
        
        if score > best_score:
            best_score = score
            best_points = points
    
    # Phase 2: Refine in general parameter space (40 parameters) from best spiral result
    if time.time() < deadline - 60.0:
        # Convert best spiral params to general params
        # Extract radii and angles from best_points
        centroid = np.mean(best_points, axis=0)
        centered = best_points - centroid
        radii = np.linalg.norm(centered, axis=1)
        angles = np.arctan2(centered[:, 1], centered[:, 0])
        
        # Ensure angles are in [0, 2*pi) and sorted
        angles = angles % (2.0 * np.pi)
        
        # Compute gaps
        gaps = np.diff(np.concatenate([angles, [angles[0] + 2.0 * np.pi]]))
        gaps = np.maximum(gaps, 1e-6)
        
        # Convert to log params
        log_radii = np.log(np.maximum(radii, 0.1))
        log_gaps = np.log(gaps)
        
        init_general = np.concatenate([log_radii, log_gaps])
        
        # Local optimization in general space
        try:
            res = minimize(
                _objective_general,
                init_general,
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-7, 'maxiter': 1000, 'maxfev': 5000}
            )
        except Exception:
            pass
        else:
            points = _general_to_points(res.x)
            score = _polygon_score(points)
            if score > best_score:
                best_score = score
                best_points = points
    
    # Phase 3: More random restarts in general space if time permits
    if time.time() < deadline - 20.0:
        base_log_gap = np.log(2.0 * np.pi / n)
        
        for i in range(10):
            if time.time() > deadline - 15.0:
                break
            
            # Generate random initial params
            log_radii_init = np.random.normal(0, 0.3, n)
            log_gaps_init = np.random.normal(base_log_gap, 0.3, n)
            
            init_params = np.concatenate([log_radii_init, log_gaps_init])
            
            try:
                res = minimize(
                    _objective_general,
                    init_params,
                    method='Nelder-Mead',
                    options={'xatol': 1e-6, 'fatol': 1e-7, 'maxiter': 500, 'maxfev': 2000}
                )
            except Exception:
                continue
                
            r_opt = res.x
            points = _general_to_points(r_opt)
            score = _polygon_score(points)
            
            if score > best_score:
                best_score = score
                best_points = points
    
    return best_points

# EVOLVE-BLOCK-END
