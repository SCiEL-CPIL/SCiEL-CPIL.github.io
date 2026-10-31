"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools


def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    # Sort points by x, then y
    pts = np.array(points)
    if len(pts) < 3:
        return 0.0
    
    # Andrew's monotone chain algorithm
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return 0.0
    
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = np.array(lower[:-1] + upper[:-1])
    
    if len(hull) < 3:
        return 0.0
    
    # Shoelace formula
    x = hull[:, 0]
    y = hull[:, 1]
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return area


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = np.inf
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                                  (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    
    return min_area


def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def construct():
    # Base grid: 4x4 grid minus one corner, centered at origin
    # 4x4 grid from -1.5 to 1.5 in steps of 1
    base_points = []
    for i in range(4):
        for j in range(4):
            x = -1.5 + i
            y = -1.5 + j
            # Skip one corner, e.g., (1.5, 1.5)
            if i == 3 and j == 3:
                continue
            base_points.append([x, y])
    
    base_points = np.array(base_points)
    
    # Apply affine transformation: M = [[a, b], [c, d]]
    # We optimize a, b, c, d to maximize min_triangle_area / hull_area
    # Since both scale linearly with det(M), the ratio is invariant to uniform scaling
    # but depends on the shape (shear, anisotropic scaling)
    
    def objective(params):
        a, b, c, d = params
        M = np.array([[a, b], [c, d]])
        transformed = base_points @ M.T
        
        hull_area = _convex_hull_area(transformed)
        if hull_area < 1e-12:
            return 0.0
        
        min_tri = _min_triangle_area(transformed)
        return -min_tri / hull_area  # Negate for minimization
    
    # Try multiple initial guesses for the affine parameters
    best_score = 0
    best_points = None
    
    initial_guesses = [
        [1.0, 0.0, 0.0, 1.0],  # Identity
        [1.0, 0.5, 0.0, 1.0],  # Shear x
        [1.0, 0.0, 0.5, 1.0],  # Shear y
        [1.0, 0.3, 0.3, 1.0],  # Both shears
        [1.2, 0.0, 0.0, 0.8],  # Anisotropic scale
        [0.8, 0.0, 0.0, 1.2],  # Anisotropic scale other way
        [1.0, 0.5, 0.5, 1.0],  # Larger shears
        [1.0, -0.3, 0.3, 1.0], # Opposite shears
    ]
    
    for guess in initial_guesses:
        try:
            result = minimize(objective, guess, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.success or result.fun < 0:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    a, b, c, d = result.x
                    M = np.array([[a, b], [c, d]])
                    best_points = base_points @ M.T
        except Exception:
            pass
    
    # If optimization didn't find anything good, use identity
    if best_points is None:
        best_points = base_points.copy()
    
    # Normalize: center at origin and scale to unit area
    # First center
    centroid = np.mean(best_points, axis=0)
    best_points = best_points - centroid
    
    # Scale to unit area
    hull_area = _convex_hull_area(best_points)
    if hull_area > 1e-12:
        scale = 1.0 / np.sqrt(hull_area)
        best_points = best_points * scale
    
    return best_points
# EVOLVE-BLOCK-END
