# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

_START_TIME = time.time()
_DEADLINE = _START_TIME + 590.0  # 10s safety margin before 600s limit


def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by triples of points."""
    n = len(points)
    if n < 3:
        return 0.0
    indices = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute A_min / A_hull."""
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    if a_hull <= 0:
        return 0.0
    return a_min / a_hull


def _soft_min_triangle_area(points, beta=50.0):
    """
    Differentiable surrogate for the minimum triangle area using log-sum-exp.
    Returns a value that approximates min(areas) when beta is large.
    """
    n = len(points)
    indices = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Log-sum-exp for soft minimum
    # soft_min = -log(sum(exp(-beta * areas))) / beta
    # For numerical stability:
    min_area = np.min(areas)
    if min_area <= 0:
        return 0.0
    exp_vals = np.exp(-beta * (areas - min_area))
    log_sum = np.log(np.sum(exp_vals))
    soft_min = min_area - log_sum / beta
    return soft_min


def _objective_soft(x, beta=50.0):
    """
    Objective to minimize: -soft_min_triangle_area / hull_area
    We want to maximize A_min / A_hull, so minimize the negative.
    """
    points = x.reshape(-1, 2)
    a_hull = _hull_area(points)
    if a_hull <= 0:
        return 1e10
    soft_min = _soft_min_triangle_area(points, beta)
    if soft_min <= 0:
        return 1e10
    return -soft_min / a_hull


def _objective_exact(x):
    """
    Objective to minimize: -A_min / A_hull
    """
    points = x.reshape(-1, 2)
    a_hull = _hull_area(points)
    if a_hull <= 0:
        return 1e10
    a_min = _min_triangle_area(points)
    if a_min <= 0:
        return 1e10
    return -a_min / a_hull


def _polish(points, max_iter=1000, tol=1e-12):
    """
    High-precision coordinate polish using L-BFGS-B on the exact objective.
    """
    x0 = points.flatten()
    result = minimize(
        _objective_exact,
        x0,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 50}
    )
    return result.x.reshape(-1, 2)


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region to maximize
    the area of the smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    best_points = None
    best_score = -np.inf
    
    # Precompute triangle indices for reuse
    tri_indices = np.array(list(itertools.combinations(range(n), 3)))
    
    # Try multiple random initializations with the soft-min surrogate approach
    rng = np.random.default_rng(seed=42)
    
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > _DEADLINE - 60:
            break
        
        # Generate random initial points in [0, 1]^2
        points_init = rng.random((n, 2))
        x0 = points_init.flatten()
        
        # Phase 1: Optimize with soft-min surrogate (differentiable)
        try:
            result_soft = minimize(
                _objective_soft,
                x0,
                method='L-BFGS-B',
                args=(30.0,),  # beta=30 for smoother optimization
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            points_soft = result_soft.x.reshape(-1, 2)
        except Exception:
            points_soft = points_init
        
        # Phase 2: Polish with exact objective
        if time.time() > _DEADLINE - 30:
            break
        
        try:
            points_polished = _polish(points_soft, max_iter=500, tol=1e-12)
        except Exception:
            points_polished = points_soft
        
        # Evaluate the score
        score = _score(points_polished)
        if score > best_score:
            best_score = score
            best_points = points_polished.copy()
    
    # If no valid result found, fall back to a simple deterministic arrangement
    if best_points is None:
        # Use a regular grid-like pattern
        best_points = np.array([
            [0.1, 0.1], [0.5, 0.1], [0.9, 0.1],
            [0.1, 0.5], [0.5, 0.5], [0.9, 0.5],
            [0.1, 0.9], [0.5, 0.9], [0.9, 0.9],
            [0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7]
        ], dtype=np.float64)
    
    # Final verification and cleanup
    best_points = np.asarray(best_points, dtype=np.float64)
    if not np.all(np.isfinite(best_points)):
        best_points = np.random.default_rng(seed=0).random((n, 2))
    
    return best_points

# EVOLVE-BLOCK-END
