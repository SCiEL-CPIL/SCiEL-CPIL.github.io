# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import itertools

_START_TIME = time.time()
_DEADLINE = _START_TIME + 590.0  # Safety margin before 600s limit

def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Generate all combinations of 3 indices
    indices = np.array(list(itertools.combinations(range(n), 3)))
    
    # Extract points
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    # Calculate areas: 0.5 * | (p2-p1) x (p3-p1) |
    v1 = p2 - p1
    v2 = p3 - p1
    # Cross product in 2D: v1_x * v2_y - v1_y * v2_x
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _soft_min_triangle_area(points, beta=50.0):
    """
    Soft-min of triangle areas using log-sum-exp (smooth surrogate for min).
    As beta -> inf, this approaches the true min.
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
    
    # Soft min: -(1/beta) * log(sum(exp(-beta * areas)))
    # Use log-sum-exp trick for numerical stability
    max_area = np.max(areas)
    # If max_area is 0, return 0
    if max_area < 1e-12:
        return 0.0
        
    shifted = -beta * (areas - max_area)
    log_sum = np.log(np.sum(np.exp(shifted))) - beta * max_area
    soft_min = -(1.0 / beta) * log_sum
    
    return soft_min

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points, beta=0.0):
    """
    Score = A_min / A_hull.
    If beta > 0, uses soft min for A_min.
    """
    a_hull = _hull_area(points)
    if a_hull <= 1e-10:
        return 0.0
    
    if beta > 0:
        a_min = _soft_min_triangle_area(points, beta)
    else:
        a_min = _min_triangle_area(points)
        
    if a_min <= 1e-10:
        return 0.0
        
    return a_min / a_hull

def _objective(x, beta=0.0):
    """
    Objective function to minimize (negative of the score).
    """
    points = x.reshape(13, 2)
    # Ensure points are not all identical or collinear in a way that breaks hull
    # A simple check: if hull area is 0, return large penalty
    score = _score(points, beta)
    if score == 0.0:
        return 1.0 # Large penalty
    return -score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the area of the
    smallest triangle formed by these points relative to the convex hull area.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    bounds = [(-1.0, 1.0)] * (2 * n)
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Differential Evolution for global search
    # Using a moderate beta for the soft min to guide the global search
    beta_global = 20.0
    
    try:
        de_result = differential_evolution(
            lambda x: _objective(x, beta_global),
            bounds,
            seed=42,
            maxiter=500,
            popsize=30,
            tol=1e-5,
            polish=False, # We will do polishing later
            init='sobol'
        )
        
        if de_result.success or de_result.fun < 0:
            best_points = de_result.x.reshape(n, 2)
            best_score = -de_result.fun
    except Exception:
        pass
    
    # If DE failed or didn't find anything good, try a few random starts with local opt
    if best_score < 0.01:
        rng = np.random.default_rng(seed=42)
        for i in range(10):
            if time.time() > _DEADLINE:
                break
            x0 = rng.uniform(-1, 1, size=(n, 2))
            try:
                res = minimize(
                    lambda x: _objective(x, beta_global),
                    x0.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200}
                )
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(n, 2)
            except Exception:
                continue
    
    if best_points is None:
        return np.random.default_rng(42).uniform(-1, 1, size=(n, 2))

    # Phase 2: Local Refinement with L-BFGS-B and higher beta
    # Increase beta to make the soft min closer to the hard min
    beta_local = 50.0
    if time.time() < _DEADLINE - 60:
        try:
            res2 = minimize(
                lambda x: _objective(x, beta_local),
                best_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            score2 = -res2.fun
            if score2 > best_score:
                best_score = score2
                best_points = res2.x.reshape(n, 2)
        except Exception:
            pass

    # Phase 3: Final Polish with Nelder-Mead on the HARD objective (beta=0)
    # This ensures we are not biased by the soft approximation
    if time.time() < _DEADLINE - 30:
        try:
            res3 = minimize(
                lambda x: _objective(x, 0.0),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            score3 = -res3.fun
            if score3 > best_score:
                best_score = score3
                best_points = res3.x.reshape(n, 2)
        except Exception:
            pass
            
    # Optional: One more very short L-BFGS-B run on hard objective if time permits
    if time.time() < _DEADLINE - 10:
        try:
            res4 = minimize(
                lambda x: _objective(x, 0.0),
                best_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14}
            )
            score4 = -res4.fun
            if score4 > best_score:
                best_score = score4
                best_points = res4.x.reshape(n, 2)
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
