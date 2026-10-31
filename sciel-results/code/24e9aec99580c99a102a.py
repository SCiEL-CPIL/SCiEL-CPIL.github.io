"""Heilbronn convex regions: 13 points, maximize min triangle area / hull area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 13


def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = np.inf
    for i, j, k in combinations(range(n), 3):
        v1 = points[j] - points[i]
        v2 = points[k] - points[i]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        area = 0.5 * abs(cross)
        if area < min_area:
            min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, 'volume' is the area
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _grid_points_3x5():
    """Generate 13 points from a 3x5 grid with one point removed (center of bottom row)."""
    # 3 rows x 5 columns = 15 points, remove 2 to get 13
    # Use a 3x5 grid: rows 0,1,2 and cols 0,1,2,3,4
    # Remove the two corner points of the middle row? Or use a different removal.
    # Let's use all 15 and remove 2 points that are least useful.
    # Actually, let's just use 13 of the 15. Remove (1,0) and (1,4) - middle row ends.
    points = []
    for r in range(3):
        for c in range(5):
            if (r == 1 and (c == 0 or c == 4)):
                continue
            points.append([float(c), float(r)])
    return np.array(points, dtype=float)


def _affine_transform(points, a, b, c, d):
    """Apply affine transformation: [a b; c d] @ [x; y]"""
    transformed = np.column_stack([
        a * points[:, 0] + b * points[:, 1],
        c * points[:, 0] + d * points[:, 1]
    ])
    return transformed


def _objective(params):
    """Objective for optimization: negative score (to minimize)."""
    a, b, c, d = params
    base = _grid_points_3x5()
    transformed = _affine_transform(base, a, b, c, d)
    # Check for degenerate (collinear) points
    if np.any(np.abs(transformed) > 1e10):
        return 1e10
    score = _score(transformed)
    return -score


def construct():
    """Construct 13 points to maximize min triangle area / convex hull area."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Base grid points (3x5 minus 2)
    base = _grid_points_3x5()
    
    # Initial guess: identity transform
    best_params = np.array([1.0, 0.0, 0.0, 1.0])
    best_score = _score(base)
    
    # Differential evolution over the 4 affine parameters
    # Search space: a, b, c, d in [-2, 2]
    bounds = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
    
    def time_check():
        return time.time() > deadline - 10.0
    
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            callback=lambda xk, convergence: None  # We'll check time separately
        )
        
        if time.time() < deadline - 5.0:
            candidate_params = result.x
            candidate_points = _affine_transform(base, *candidate_params)
            candidate_score = _score(candidate_points)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_params = candidate_params.copy()
                
    except Exception:
        pass
    
    # Local refinement with Nelder-Mead
    if time.time() < deadline - 5.0:
        try:
            result2 = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if time.time() < deadline - 2.0:
                candidate_params = result2.x
                candidate_points = _affine_transform(base, *candidate_params)
                candidate_score = _score(candidate_points)
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_params = candidate_params.copy()
                    
        except Exception:
            pass
    
    # Apply the best affine transform to the base grid
    final_points = _affine_transform(base, *best_params)
    
    return final_points
