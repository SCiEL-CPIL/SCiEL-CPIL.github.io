"""Heilbronn convex regions: maximize minimum triangle area / convex hull area for 13 points."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13
DEADLINE_MARGIN = 10.0  # seconds to leave before the 150s wall-clock kill

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area of triangle = 0.5 * |cross product|
                ax = p2[0] - p1[0]
                ay = p2[1] - p1[1]
                bx = p3[0] - p1[0]
                by = p3[1] - p1[1]
                area = 0.5 * abs(ax * by - ay * bx)
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _negative_score(points):
    """Negative of the score for minimization."""
    return -_score(points)

def _regular_polygon(n, center=(0, 0), radius=1.0):
    """Generate a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])

def construct():
    start_time = time.time()
    
    # Start with a regular 13-gon
    best_points = _regular_polygon(N, radius=1.0)
    best_score = _score(best_points)
    
    # Use Nelder-Mead optimization to improve the configuration
    # We'll try multiple restarts with slight perturbations
    
    restarts = [
        # Regular polygon
        _regular_polygon(N, radius=1.0),
        # Slightly perturbed regular polygon
        _regular_polygon(N, radius=1.0) + np.random.RandomState(42).normal(0, 0.01, (N, 2)),
        # Different rotation
        _regular_polygon(N, radius=1.0, center=(0, 0)),
    ]
    
    for init_points in restarts:
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        
        try:
            result = minimize(
                _negative_score,
                init_points,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if result.success or result.fun < -best_score:
                current_score = -result.fun
                if current_score > best_score:
                    best_score = current_score
                    best_points = result.x.copy()
        except:
            continue
    
    # If we have time, try a few more random restarts
    remaining_time = 150.0 - DEADLINE_MARGIN - (time.time() - start_time)
    if remaining_time > 5:
        rng = np.random.RandomState(123)
        max_extra_restarts = int(remaining_time / 5)  # ~5 seconds per restart
        for _ in range(max_extra_restarts):
            if time.time() - start_time > 150.0 - DEADLINE_MARGIN:
                break
            # Random points in unit circle
            r = np.sqrt(rng.uniform(0, 1, N))
            theta = rng.uniform(0, 2 * np.pi, N)
            init_points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            
            try:
                result = minimize(
                    _negative_score,
                    init_points,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                )
                
                if result.fun < -best_score:
                    current_score = -result.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_points = result.x.copy()
            except:
                continue
    
    return best_points
