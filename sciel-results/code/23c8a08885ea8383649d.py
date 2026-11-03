"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms

def _covering_radius(points):
    """
    Compute the covering radius of the set of points on the unit sphere.
    This is the maximum circumradius of the facets of the convex hull.
    """
    try:
        hull = ConvexHull(points)
        max_radius = 0.0
        for simplex in hull.simplices:
            # simplex is an array of indices of the points forming the facet
            # The circumradius of a simplex can be computed
            # For a triangle (2D facet in 3D), the circumradius R is given by:
            # R = abc / (4 * Area)
            # But we can also use the formula for the distance from the centroid to the vertices? No, circumcenter.
            # A robust way is to solve for the circumcenter.
            # However, for small N, we can use a simpler approximation or a specific formula.
            # Let's use the formula for the circumradius of a triangle.
            # If the facet is not a triangle (which it should be in 3D for a convex hull of points in general position),
            # we might have issues. But ConvexHull returns simplices which are triangles in 3D.
            
            p0 = points[simplex[0]]
            p1 = points[simplex[1]]
            p2 = points[simplex[2]]
            
            # Side lengths
            a = np.linalg.norm(p1 - p2)
            b = np.linalg.norm(p0 - p2)
            c = np.linalg.norm(p0 - p1)
            
            # Area of the triangle
            # Area = 0.5 * | (p1 - p0) x (p2 - p0) |
            v1 = p1 - p0
            v2 = p2 - p0
            cross = np.cross(v1, v2)
            area = 0.5 * np.linalg.norm(cross)
            
            if area < 1e-10:
                continue
                
            # Circumradius R = abc / (4 * Area)
            R = (a * b * c) / (4.0 * area)
            if R > max_radius:
                max_radius = R
                
        return max_radius
    except Exception:
        # If hull computation fails, return a large value
        return 10.0

def _local_optimize_covering(points, n_iter=5000, step_size=0.1):
    """
    Local optimization to minimize the covering radius.
    We try to move each point slightly to reduce the covering radius.
    """
    points = _normalize(points).copy()
    current_radius = _covering_radius(points)
    best_points = points.copy()
    best_radius = current_radius
    
    for i in range(n_iter):
        # Decrease step size over time
        current_step = step_size * (1.0 - i / n_iter)
        
        # Select a random point
        idx = np.random.randint(N)
        original_point = points[idx].copy()
        
        # Generate a random direction
        direction = np.random.normal(0, 1, 3)
        direction = direction / np.linalg.norm(direction)
        
        # Move the point
        new_point = original_point + current_step * direction
        new_point = new_point / np.linalg.norm(new_point)
        
        points[idx] = new_point
        new_radius = _covering_radius(points)
        
        if new_radius < current_radius:
            current_radius = new_radius
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = points.copy()
        else:
            # Revert
            points[idx] = original_point
            
    return best_points

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    # 1. Initialize with Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Local Optimization on Covering Radius
    # We run multiple iterations to escape local minima
    points_final = x0
    for _ in range(3):
        points_final = _local_optimize_covering(points_final, n_iter=2000, step_size=0.2)
        
    # 3. Final fine-tuning
    points_final = _local_optimize_covering(points_final, n_iter=2000, step_size=0.05)
    
    return _normalize(points_final)
# EVOLVE-BLOCK-END
