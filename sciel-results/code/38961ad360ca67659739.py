# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import itertools

_START_TIME = time.time()
_DEADLINE = _START_TIME + 575.0  # Safety margin before 630s limit

def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    Vectorized for speed.
    """
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
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Score = A_min / A_hull
    """
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    
    if a_hull <= 1e-10 or a_min <= 1e-10:
        return 0.0
    
    return a_min / a_hull

def _objective(x):
    """
    Exact objective (non-smooth). Returns negative score for minimization.
    """
    points = x.reshape(13, 2)
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    
    if a_hull <= 1e-10 or a_min <= 1e-10:
        return 0.0
    
    return -(a_min / a_hull)

def _make_regular_polygon_center(n_outer=12):
    """
    Create a regular polygon with n_outer points plus a center point.
    This is a strong geometric starting point for the Heilbronn problem.
    """
    points = [[0.0, 0.0]]  # center
    for k in range(n_outer):
        angle = 2 * np.pi * k / n_outer
        x = np.cos(angle)
        y = np.sin(angle)
        points.append([x, y])
    return np.array(points)

def _de_callback(xk, convergence):
    """
    Callback for differential evolution to check time limit.
    """
    if time.time() > _DEADLINE - 30:
        raise KeyboardInterrupt("Time limit approaching")

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the area of the
    smallest triangle formed by these points relative to the convex hull area.
    
    Uses differential evolution for global search followed by local refinement.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(seed=42)
    
    # Start with the regular 12-gon + center configuration
    # This is known to be a strong starting point
    initial_pts = _make_regular_polygon_center(12)
    initial_score = _score(initial_pts)
    
    if initial_score > best_score:
        best_score = initial_score
        best_points = initial_pts.copy()
    
    # Define bounds for differential evolution
    # Points should be in a reasonable range
    bounds = [(-2, 2)] * (n * 2)
    
    # Phase 1: Differential evolution for global optimization
    try:
        de_result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            mutation=(0.3, 1.0),
            recombination=0.7,
            callback=_de_callback,
            polish=False,  # We'll do our own polishing
            init='sobol'
        )
        
        if de_result.success or de_result.fun < _objective(best_points.flatten()):
            de_pts = de_result.x.reshape(n, 2)
            de_score = _score(de_pts)
            
            if de_score > best_score:
                best_score = de_score
                best_points = de_pts.copy()
                
    except KeyboardInterrupt:
        pass
    except Exception:
        pass
    
    # Phase 2: Local refinement using Nelder-Mead from multiple starting points
    # Use the best DE result and the initial regular polygon
    start_candidates = []
    
    if best_points is not None:
        start_candidates.append(best_points.copy())
    
    # Add the regular polygon as another candidate
    start_candidates.append(_make_regular_polygon_center(12))
    
    # Add a slightly perturbed version
    perturbed = _make_regular_polygon_center(12) + rng.normal(0, 0.1, (n, 2))
    start_candidates.append(perturbed)
    
    for start_pts in start_candidates:
        if time.time() > _DEADLINE - 10:
            break
        
        try:
            res = minimize(
                _objective,
                start_pts.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            current_pts = res.x.reshape(n, 2)
            current_score = _score(current_pts)
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_pts.copy()
                
        except Exception:
            continue
    
    # Phase 3: If time permits, try a few more local refinements from random perturbations
    remaining_time = _DEADLINE - time.time()
    if remaining_time > 30 and best_points is not None:
        num_extra = min(5, int(remaining_time / 10))
        
        for i in range(num_extra):
            if time.time() > _DEADLINE - 5:
                break
            
            # Perturb the best solution
            perturbed = best_points + rng.normal(0, 0.05, (n, 2))
            
            try:
                res = minimize(
                    _objective,
                    perturbed.flatten(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                
                current_pts = res.x.reshape(n, 2)
                current_score = _score(current_pts)
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_pts.copy()
                    
            except Exception:
                continue
    
    if best_points is None:
        best_points = rng.random((n, 2))
    
    return best_points

# EVOLVE-BLOCK-END
