"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

def _min_triangle_area_vec(points):
    """Compute the minimum area of any triangle formed by 3 points (vectorized)."""
    n = len(points)
    if n < 3:
        return 0.0
    indices = np.array(list(combinations(range(n), 3)))
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    
    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # Fallback: simple method using shoelace on sorted points
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
    min_area = _min_triangle_area_vec(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return -min_area / hull_area

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    min_area = _min_triangle_area_vec(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def construct():
    # Global search using Differential Evolution
    bounds = [(-10, 10)] * (N * 2)
    
    de_result = differential_evolution(
        _objective,
        bounds,
        maxiter=50,
        popsize=15,
        tol=1e-6,
        seed=42,
        polish=False
    )
    
    best_points = de_result.x.reshape(N, 2)
    best_score = _score(best_points)
    
    # Local polish using Nelder-Mead
    nm_result = minimize(
        _objective,
        best_points.flatten(),
        method='Nelder-Mead',
        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
    )
    nm_points = nm_result.x.reshape(N, 2)
    nm_score = _score(nm_points)
    
    if nm_score > best_score:
        best_points = nm_points
        best_score = nm_score
        
    # Also try polishing a regular polygon start just in case DE missed it
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_points = np.column_stack([np.cos(angles), np.sin(angles)])
    reg_score = _score(reg_points)
    
    if reg_score > best_score:
        best_points = reg_points
        best_score = reg_score
        
    nm_reg = minimize(
        _objective,
        reg_points.flatten(),
        method='Nelder-Mead',
        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
    )
    nm_reg_points = nm_reg.x.reshape(N, 2)
    nm_reg_score = _score(nm_reg_points)
    
    if nm_reg_score > best_score:
        best_points = nm_reg_points
        best_score = nm_reg_score

    return best_points
# EVOLVE-BLOCK-END
