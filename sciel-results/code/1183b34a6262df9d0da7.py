"""Heilbronn convex regions - optimized placement of 13 points."""
import numpy as np
from scipy.optimize import minimize
import time

N = 13

def _triangle_area(p1, p2, p3):
    """Area of triangle with vertices p1, p2, p3 (2D points)."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _convex_hull_area(points):
    """Compute area of convex hull of points using scipy."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use polygon area if all points are in convex position
        # For a regular polygon or near-regular, we can compute directly
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1] - points[j][0] * points[i][1]
        return abs(area) * 0.5

def _min_triangle_area(points):
    """Compute minimum triangle area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _regular_polygon(n, center=None, radius=1.0):
    """Generate vertices of a regular n-gon."""
    if center is None:
        center = np.array([0.0, 0.0])
    angles = 2 * np.pi * np.arange(n) / n
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack((x, y))

def _optimize_from_regular(n, time_limit=120):
    """Optimize point positions starting from a regular polygon."""
    start_time = time.time()
    
    # Start with regular polygon
    points = _regular_polygon(n)
    current_score = _score(points)
    best_points = points.copy()
    best_score = current_score
    
    # Try to optimize using angular perturbations
    # Parameterize: each point's angle can be slightly adjusted
    # Use Nelder-Mead on the angles
    
    def objective(params):
        if time.time() - start_time > time_limit - 5:
            return -best_score  # Return best known to stop early
        
        angles = 2 * np.pi * np.arange(n) / n + params
        x = np.cos(angles)
        y = np.sin(angles)
        pts = np.column_stack((x, y))
        return -_score(pts)  # Minimize negative score
    
    # Initial perturbation
    x0 = np.zeros(n)
    
    # Use Nelder-Mead optimization
    try:
        result = minimize(
            objective, x0, method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        if result.fun < -best_score:
            best_score = -result.fun
            angles = 2 * np.pi * np.arange(n) / n + result.x
            best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    except Exception:
        pass
    
    # Try radial perturbations too
    def objective_radial(params):
        if time.time() - start_time > time_limit - 5:
            return -best_score
        
        angles = 2 * np.pi * np.arange(n) / n
        radii = 1.0 + params
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        pts = np.column_stack((x, y))
        return -_score(pts)
    
    x0_radial = np.zeros(n)
    try:
        result2 = minimize(
            objective_radial, x0_radial, method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        if result2.fun < -best_score:
            best_score = -result2.fun
            angles = 2 * np.pi * np.arange(n) / n
            radii = 1.0 + result2.x
            best_points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    except Exception:
        pass
    
    return best_points

def construct():
    """Place 13 points to maximize min triangle area / convex hull area."""
    start_time = time.time()
    time_limit = 120  # Safety margin
    
    # Start with regular polygon
    points = _regular_polygon(N)
    current_score = _score(points)
    best_points = points.copy()
    best_score = current_score
    
    # Try optimization if we have time
    elapsed = time.time() - start_time
    if elapsed < time_limit - 30:
        try:
            optimized = _optimize_from_regular(N, time_limit=time_limit - elapsed)
            opt_score = _score(optimized)
            if opt_score > best_score:
                best_points = optimized
                best_score = opt_score
        except Exception:
            pass
    
    return best_points
