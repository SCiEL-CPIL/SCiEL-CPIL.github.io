"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Twice the signed area
                area2 = abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                if area2 < min_area:
                    min_area = area2
    return min_area / 2.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 1e-10:
            return 0.0
        min_tri = _min_triangle_area(points)
        return min_tri / hull_area
    except Exception:
        return 0.0

def _regular_polygon(n, radius=1.0):
    """Generate points of a regular n-gon centered at origin."""
    angles = 2 * np.pi * np.arange(n) / n
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    best_points = None
    best_score = 0.0
    
    # Start with regular 13-gon
    points = _regular_polygon(N, radius=1.0)
    best_points = points.copy()
    best_score = _score(points)
    
    # Define the objective to minimize (negative score)
    def neg_score(flat_points):
        pts = flat_points.reshape(N, 2)
        return -_score(pts)
    
    # Nelder-Mead optimization
    x0 = points.flatten()
    
    # Try multiple local optimizations from different starting points
    starts = [x0.copy()]
    
    # Add some perturbed versions
    rng = np.random.RandomState(42)
    for _ in range(5):
        perturbed = x0 + rng.normal(0, 0.05, size=x0.shape)
        starts.append(perturbed)
    
    for x_start in starts:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                neg_score,
                x_start,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if result.fun < -best_score:
                best_score = -result.fun
                best_points = result.x.reshape(N, 2)
        except Exception:
            pass
    
    # Second round: more aggressive local search from best point
    if time.time() < deadline - 30:
        x0 = best_points.flatten()
        for _ in range(3):
            if time.time() > deadline:
                break
            try:
                result = minimize(
                    neg_score,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 5000,
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True
                    }
                )
                if result.fun < -best_score:
                    best_score = -result.fun
                    best_points = result.x.reshape(N, 2)
                    x0 = best_points.flatten()
            except Exception:
                pass
    
    return best_points
# EVOLVE-BLOCK-END
