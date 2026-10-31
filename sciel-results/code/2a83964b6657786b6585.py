"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _regular_polygon_points(n, radius=1.0, center=(0.0, 0.0)):
    """Generate n points on a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])


def _structured_initial(n, variant=0):
    """Generate structured initial configurations for n points."""
    if variant == 0:
        # Regular polygon
        return _regular_polygon_points(n, radius=1.0)
    elif variant == 1:
        # Two concentric rings
        outer = _regular_polygon_points(7, radius=1.0)
        inner = _regular_polygon_points(6, radius=0.5)
        return np.vstack([outer, inner])
    elif variant == 2:
        # Three concentric rings
        outer = _regular_polygon_points(5, radius=1.0)
        mid = _regular_polygon_points(5, radius=0.6)
        inner = _regular_polygon_points(3, radius=0.25)
        return np.vstack([outer, mid, inner])
    elif variant == 3:
        # Grid-like arrangement
        pts = []
        # Center
        pts.append([0.0, 0.0])
        # Ring of 6
        for i in range(6):
            angle = i * np.pi / 3
            pts.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
        # Outer ring of 6
        for i in range(6):
            angle = (i + 0.5) * np.pi / 3
            pts.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
        return np.array(pts)
    elif variant == 4:
        # Slightly perturbed regular polygon
        pts = _regular_polygon_points(n, radius=1.0)
        rng = np.random.RandomState(variant)
        pts += 0.05 * rng.randn(n, 2)
        return pts
    else:
        return _regular_polygon_points(n, radius=1.0)


def _optimize_from_initial(points, deadline, max_iter=2000, tol=1e-10):
    """Optimize points to maximize the score using Nelder-Mead."""
    start_time = time.time()
    
    def objective(x):
        pts = x.reshape(-1, 2)
        return -_score(pts)  # Negate for minimization
    
    result = minimize(
        objective,
        points.flatten(),
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': tol, 'adaptive': True}
    )
    
    elapsed = time.time() - start_time
    return result.x.reshape(-1, 2), _score(result.x.reshape(-1, 2))


def construct():
    deadline = time.time() + 110  # 10s safety margin
    
    best_points = None
    best_score = 0.0
    
    # Try multiple initial configurations
    for variant in range(5):
        if time.time() > deadline:
            break
        
        initial = _structured_initial(N, variant=variant)
        
        # Optimize from this initial configuration
        try:
            optimized, score = _optimize_from_initial(initial, deadline, max_iter=1500)
            if score > best_score:
                best_score = score
                best_points = optimized.copy()
        except Exception:
            continue
    
    # If no optimization succeeded, return the best initial configuration
    if best_points is None:
        best_points = _structured_initial(N, variant=0)
    
    return best_points
# EVOLVE-BLOCK-END
