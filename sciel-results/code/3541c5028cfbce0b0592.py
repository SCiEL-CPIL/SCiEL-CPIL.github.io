"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area of triangle with vertices points[i], points[j], points[k]
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # For 2D, volume is area
    except Exception:
        # Fallback: simple polygon area if all points are on hull
        # This shouldn't happen for 14 random points, but just in case
        return 1.0


def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area


def construct():
    """
    Use differential evolution to find 14 points in the unit square that
    maximize the minimum triangle area relative to the convex hull area.
    """
    # Define bounds: each point is in [0, 1] x [0, 1]
    bounds = [(0, 1)] * (2 * N)
    
    def objective(x):
        points = x.reshape(N, 2)
        # We want to maximize score, so return negative for minimization
        return -_score(points)
    
    # Run differential evolution
    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=100,
        popsize=15,
        tol=1e-4,
        seed=42,
        polish=False
    )
    
    best_points = result.x.reshape(N, 2)
    
    # Optional: try to improve with local refinement using Nelder-Mead
    from scipy.optimize import minimize
    
    # Try a few local refinements from the best solution
    best_score = _score(best_points)
    best_points_refined = best_points.copy()
    
    for attempt in range(5):
        res = minimize(
            lambda x: -_score(x.reshape(N, 2)),
            best_points_refined.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if res.fun < -best_score:
            best_points_refined = res.x.reshape(N, 2)
            best_score = -res.fun
    
    return best_points_refined
# EVOLVE-BLOCK-END
