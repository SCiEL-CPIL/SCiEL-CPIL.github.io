"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area of triangle = 0.5 * |cross product|
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback for degenerate cases
        return 0.0

def _objective(x):
    """Objective to minimize: negative of (min_triangle_area / hull_area)."""
    points = x.reshape(N, 2)
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return -min_area / hull_area

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def construct():
    # Define bounds: points in [0, 1] x [0, 1]
    bounds = [(0, 1) for _ in range(N * 2)]
    
    # Use Differential Evolution for global optimization
    # This is a population-based stochastic global search, distinct from local Nelder-Mead
    result = differential_evolution(
        _objective,
        bounds,
        seed=42,
        maxiter=200,
        popsize=20,
        tol=1e-6,
        polish=True,  # Use Nelder-Mead for final polish
        init='sobol'  # Sobol sequence for better initial population distribution
    )
    
    best_points = result.x.reshape(N, 2)
    
    # Verify score
    final_score = _score(best_points)
    
    # If DE fails or yields poor results, fallback to a simple heuristic
    if final_score < 0.05:
        # Fallback: Regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
# EVOLVE-BLOCK-END
