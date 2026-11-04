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
        # Fallback: simple method using shoelace on sorted points
        # Sort by angle from centroid
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        sorted_points = points[np.argsort(angles)]
        area = 0.0
        n = len(sorted_points)
        for i in range(n):
            j = (i + 1) % n
            area += sorted_points[i][0] * sorted_points[j][1]
            area -= sorted_points[j][0] * sorted_points[i][1]
        return abs(area) / 2.0

def _objective(x):
    """Objective to minimize: negative of (min_triangle_area / hull_area)."""
    points = x.reshape(N, 2)
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return -min_area / hull_area

def construct():
    # Use Differential Evolution for global optimization
    # This is more robust than Nelder-Mead for non-smooth, multi-modal landscapes.
    
    # Bounds: The problem is scale and translation invariant.
    # We use a box to constrain the search space.
    bounds = [(-5.0, 5.0) for _ in range(N * 2)]
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Run Differential Evolution
    # maxiter: 100 generations is usually sufficient for N=13
    # tol: convergence tolerance
    # polish: False, as L-BFGS-B may struggle with non-smooth objective
    result = differential_evolution(
        _objective,
        bounds,
        seed=42,
        maxiter=100,
        tol=1e-6,
        polish=False
    )
    
    best_points = result.x.reshape(N, 2)
    
    # Optional: Refine with Nelder-Mead starting from DE result
    # This can squeeze out more performance.
    try:
        from scipy.optimize import minimize
        result_refined = minimize(
            _objective,
            best_points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        refined_points = result_refined.x.reshape(N, 2)
        
        # Check if refined is better
        score_de = -_objective(best_points.flatten())
        score_refined = -_objective(result_refined.x)
        
        if score_refined > score_de:
            best_points = refined_points
    except Exception:
        pass
        
    return best_points
# EVOLVE-BLOCK-END
