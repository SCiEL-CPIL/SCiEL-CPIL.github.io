"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _objective(params):
    """Objective for differential_evolution: minimize negative score."""
    points = params.reshape(-1, 2)
    s = _score(points)
    return -s


def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Initial random points
    rng = np.random.RandomState(N)
    best_points = rng.rand(N, 2)
    best_score = _score(best_points)
    
    # Define bounds for differential evolution
    bounds = [(0, 1)] * (2 * N)
    
    # Try differential evolution with various settings
    strategies = [
        {'maxiter': 200, 'popsize': 10, 'tol': 1e-6, 'seed': 42},
        {'maxiter': 300, 'popsize': 15, 'tol': 1e-7, 'seed': 123},
        {'maxiter': 150, 'popsize': 20, 'tol': 1e-5, 'seed': 7},
    ]
    
    for attempt, kwargs in enumerate(strategies):
        if time.time() > deadline - 30:  # Need at least 30s remaining
            break
        
        try:
            result = differential_evolution(
                _objective, bounds,
                maxiter=kwargs['maxiter'],
                popsize=kwargs['popsize'],
                tol=kwargs['tol'],
                seed=kwargs['seed'],
                polish=False,
                init='latinhypercube'
            )
            
            if result.success or result.fun < 0:
                candidate_points = result.x.reshape(-1, 2)
                candidate_score = _score(candidate_points)
                
                if candidate_score > best_score:
                    best_points = candidate_points.copy()
                    best_score = candidate_score
                    
        except Exception:
            continue
        
        if time.time() > deadline - 20:
            break
    
    return best_points
# EVOLVE-BLOCK-END
