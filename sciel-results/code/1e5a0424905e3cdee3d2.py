"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0

def _score(points):
    """Calculate the ratio of min triangle area to convex hull area."""
    hull_a = _hull_area(points)
    if hull_a < 1e-9:
        return -1.0
    min_a = _min_triangle_area(points)
    return min_a / hull_a

def _is_convex(points):
    """Check if points form a strictly convex polygon in CCW order."""
    n = len(points)
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]
        cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
        if cross < -1e-9:
            return False
    return True

def construct():
    # Initialize with a regular N-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Objective function to minimize (negative of the score)
    def objective(x):
        p = x.reshape(-1, 2)
        # Penalty if not convex
        if not _is_convex(p):
            return 10.0
        s = _score(p)
        return -s

    x0 = points.flatten()
    
    # Optimize the positions
    # Nelder-Mead is robust and doesn't require gradients
    start_time = time.time()
    deadline = 100.0
    
    try:
        # Run optimization
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-6})
        
        # Check if we exceeded time limit (approximate, as minimize doesn't support callback for time easily in all versions)
        # But 5000 iterations on 28D Nelder-Mead is usually fast enough for this simple function.
        
        if result.x is not None:
            best_points = result.x.reshape(-1, 2)
            # Ensure the optimized points are valid and better
            if _is_convex(best_points) and _score(best_points) > _score(points):
                points = best_points
    except Exception:
        pass
        
    return points
# EVOLVE-BLOCK-END
