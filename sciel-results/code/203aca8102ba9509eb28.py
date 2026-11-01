"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random


def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of points."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Find the point with lowest y (and lowest x in case of tie)
    start = 0
    for i in range(1, n):
        if points[i][1] < points[start][1] or (points[i][1] == points[start][1] and points[i][0] < points[start][0]):
            start = i
    
    # Sort by polar angle
    def angle_cmp(p):
        return np.arctan2(p[1] - points[start][1], p[0] - points[start][0])
    
    sorted_pts = sorted(points, key=angle_cmp)
    sorted_pts.insert(0, points[start])  # Put start first
    
    # Andrew's monotone chain algorithm
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in sorted_pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(sorted_pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = lower[:-1] + upper[:-1]
    
    if len(hull) < 3:
        return 0.0
    
    # Shoelace formula
    area = 0.0
    m = len(hull)
    for i in range(m):
        j = (i + 1) % m
        area += hull[i][0] * hull[j][1]
        area -= hull[j][0] * hull[i][1]
    return abs(area) / 2.0


def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def construct():
    """
    Place 14 points in a planar convex region to maximize the minimum triangle area
    normalized by the convex hull area.
    
    Strategy:
    1. Start with a regular polygon or grid-based initial configuration.
    2. Use local optimization (Nelder-Mead or similar) to maximize the score.
    3. Try multiple initial configurations and keep the best.
    """
    best_points = None
    best_score = -1.0
    
    # Try several initial configurations
    initial_configs = []
    
    # Config 1: Regular 14-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Config 2: Regular 14-gon slightly perturbed
    rng = random.Random(42)
    angles2 = angles + np.array([rng.gauss(0, 0.05) for _ in range(N)])
    initial_configs.append(np.column_stack([np.cos(angles2), np.sin(angles2)]))
    
    # Config 3: Two concentric heptagons (7+7 points)
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
    outer_pts = np.column_stack([np.cos(angles_outer) * 1.0, np.sin(angles_outer) * 1.0])
    inner_pts = np.column_stack([np.cos(angles_inner) * 0.5, np.sin(angles_inner) * 0.5])
    initial_configs.append(np.vstack([outer_pts, inner_pts]))
    
    # Config 4: Grid-like arrangement
    grid_x = np.array([0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.25, 0.55, 0.75, 0.35, 0.65, 0.45, 0.55])
    grid_y = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85])
    initial_configs.append(np.column_stack([grid_x, grid_y]))
    
    # Config 5: Random points in unit square
    rng2 = random.Random(123)
    initial_configs.append(np.array([[rng2.random(), rng2.random()] for _ in range(N)]))
    
    # Config 6: Regular 14-gon with alternating radii
    angles6 = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = np.array([1.0 if i % 2 == 0 else 0.7 for i in range(N)])
    initial_configs.append(np.column_stack([np.cos(angles6) * radii, np.sin(angles6) * radii]))
    
    for init_pts in initial_configs:
        # Flatten for optimization
        x0 = init_pts.flatten()
        
        def negative_score(x):
            pts = x.reshape(N, 2)
            return -_score(pts)
        
        # Use Nelder-Mead for local optimization
        try:
            result = minimize(negative_score, x0, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            opt_pts = result.x.reshape(N, 2)
            s = -result.fun
            
            if s > best_score:
                best_score = s
                best_points = opt_pts
        except Exception:
            pass
    
    # If no optimization succeeded, fall back to the best initial config
    if best_points is None:
        for init_pts in initial_configs:
            s = _score(init_pts)
            if s > best_score:
                best_score = s
                best_points = init_pts
    
    # Final fallback: regular polygon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
