"""Heilbronn convex regions n=14: maximize minimum triangle area over convex hull area."""
import time
import numpy as np
from itertools import combinations

N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * |cross product|
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    n = len(points)
    if n < 3:
        return 0.0
    # Use scipy's ConvexHull
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use all points, compute area via shoelace after sorting by angle
        # This is a rough fallback
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]
        # Shoelace formula
        x = sorted_points[:, 0]
        y = sorted_points[:, 1]
        area = 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
        return area

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _regular_polygon(n, radius=1.0):
    """Create a regular n-gon centered at origin."""
    angles = 2 * np.pi * np.arange(n) / n
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _optimize_local(points, time_budget=30.0):
    """Local optimization using Nelder-Mead on the flattened point array."""
    start_time = time.time()
    x0 = points.flatten()
    
    def objective(x):
        if time.time() - start_time > time_budget:
            return 0.0
        pts = x.reshape(-1, 2)
        return -_score(pts)  # Negate for minimization
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
    
    if time.time() - start_time > time_budget:
        return points  # Return original if timeout
    
    optimized = result.x.reshape(-1, 2)
    return optimized

def construct():
    """Construct 14 points to maximize minimum triangle area / convex hull area."""
    start_time = time.time()
    deadline = 120.0  # Safety margin
    
    # Start with regular 14-gon
    points = _regular_polygon(N, radius=1.0)
    best_score = _score(points)
    best_points = points.copy()
    
    # Try local optimization
    if time.time() - start_time < deadline - 20:
        optimized = _optimize_local(points, time_budget=deadline - time.time() - 10)
        opt_score = _score(optimized)
        if opt_score > best_score:
            best_score = opt_score
            best_points = optimized.copy()
    
    # Try a few random restarts if time permits
    remaining = deadline - time.time()
    if remaining > 15:
        import random
        rng = random.Random(42)
        n_restarts = min(int(remaining / 5), 5)  # Up to 5 restarts, ~5s each
        for _ in range(n_restarts):
            if time.time() - start_time > deadline - 10:
                break
            # Start from a perturbed regular polygon
            perturbation = np.random.normal(0, 0.1, (N, 2))
            trial_points = _regular_polygon(N, radius=1.0) + perturbation
            trial_score = _score(trial_points)
            if trial_score > best_score:
                best_score = trial_score
                best_points = trial_points.copy()
            
            # Local optimize if time permits
            if time.time() - start_time < deadline - 15:
                opt_pts = _optimize_local(trial_points, time_budget=min(10, deadline - time.time() - 5))
                opt_score = _score(opt_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_pts.copy()
    
    # Normalize to unit area (the task says "normalized to unit area")
    hull_area = _convex_hull_area(best_points)
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        best_points = best_points * scale
    
    return best_points
# EVOLVE-BLOCK-END
