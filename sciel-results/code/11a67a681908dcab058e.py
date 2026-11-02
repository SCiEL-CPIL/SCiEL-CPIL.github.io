"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 14

# EVOLVE-BLOCK-START
def _triangle_areas(points):
    """Compute all triangle areas from a set of points. Returns array of shape (n, n, n)."""
    n = len(points)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                              (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                areas[i, j, k] = a
                areas[j, i, k] = a
                areas[k, i, j] = a
    return areas


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
    sorted_pts.insert(0, points[start])
    sorted_pts = sorted_pts[1:]  # remove duplicate start point
    
    # Build hull using Graham scan
    hull = []
    for p in sorted_pts:
        while len(hull) >= 2:
            p1 = hull[-2]
            p2 = hull[-1]
            cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
            if cross <= 1e-12:
                hull.pop()
            else:
                break
        hull.append(p)
    
    if len(hull) < 3:
        return 0.0
    
    # Compute area of convex polygon
    area = 0.0
    m = len(hull)
    for i in range(m):
        j = (i + 1) % m
        area += hull[i][0] * hull[j][1] - hull[j][0] * hull[i][1]
    return abs(area) * 0.5


def _score(points):
    """Compute the score: minimum triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area


def _objective(params, N=N):
    """Objective function for optimization: negative of the score."""
    points = params.reshape(N, 2)
    return -_score(points)


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_score = -np.inf
    best_points = None
    
    # Initial heuristic: place points on a circle with some perturbation
    # This tends to give good minimum triangle areas
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radius = 1.0
    init_points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    
    # Normalize to unit area
    hull_area = _convex_hull_area(init_points)
    if hull_area > 1e-12:
        init_points = init_points / np.sqrt(hull_area)
    
    best_points = init_points.copy()
    best_score = _score(best_points)
    
    # Try multiple random restarts with optimization
    rng = np.random.RandomState(42)
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Generate initial point set
        if restart == 0:
            points = best_points.copy()
        else:
            # Random initialization within unit square
            points = rng.random((N, 2))
        
        # Normalize
        hull_area = _convex_hull_area(points)
        if hull_area > 1e-12:
            points = points / np.sqrt(hull_area)
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(_objective, points.flatten(), method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.success or result.fun < best_score:
                opt_points = result.x.reshape(N, 2)
                opt_score = -result.fun
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    # Final polish: run more iterations on the best point set
    if time.time() < deadline - 5:
        try:
            result = minimize(_objective, best_points.flatten(), method='L-BFGS-B',
                              options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < best_score:
                opt_points = result.x.reshape(N, 2)
                opt_score = -result.fun
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
