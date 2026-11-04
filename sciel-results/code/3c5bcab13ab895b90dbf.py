N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
from itertools import combinations

def _min_triangle_area(points):
    """Compute minimum triangle area over all triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute convex hull area using scipy."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 1.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    ch_area = _convex_hull_area(points)
    if ch_area <= 0:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / ch_area

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), start_angle=0.0):
    """Generate n points on a regular polygon."""
    angles = start_angle + 2 * np.pi * np.arange(n) / n
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])

def _initial_config_14gon():
    """14 points on a regular 14-gon."""
    return _regular_polygon(14, radius=1.0)

def _initial_config_two_heptagons():
    """7 points on outer heptagon + 7 on inner heptagon (rotated by pi/7)."""
    outer = _regular_polygon(7, radius=1.0, start_angle=0.0)
    inner = _regular_polygon(7, radius=0.5, start_angle=np.pi/7)
    return np.vstack([outer, inner])

def _initial_config_grid_like():
    """A grid-like arrangement for 14 points."""
    # 4x4 grid minus 2 corners, scaled
    pts = []
    for i in range(4):
        for j in range(4):
            if not (i == 0 and j == 0) and not (i == 3 and j == 3):
                pts.append([i - 1.5, j - 1.5])
    pts = np.array(pts)
    # Center and scale
    pts = pts - pts.mean(axis=0)
    max_norm = np.max(np.linalg.norm(pts, axis=1))
    pts = pts / max_norm
    return pts

def _optimize_from_start(start_points, max_iter=50, time_budget=30):
    """Optimize from a starting configuration using Nelder-Mead."""
    start_time = time.time()
    
    def objective(flat_points):
        if time.time() - start_time > time_budget:
            return 0.0
        pts = flat_points.reshape(-1, 2)
        return -_score(pts)
    
    x0 = start_points.flatten()
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
    
    if result.success or result.fun < 0:
        return result.x.reshape(-1, 2), -result.fun
    else:
        return start_points, _score(start_points)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Try multiple starting configurations
    starts = [
        _initial_config_14gon(),
        _initial_config_two_heptagons(),
        _initial_config_grid_like(),
    ]
    
    best_points = None
    best_score = 0.0
    
    for i, start in enumerate(starts):
        if time.time() > deadline:
            break
        
        remaining_time = deadline - time.time()
        if remaining_time < 10:
            break
        
        # Allocate time budget based on remaining time
        time_budget = min(remaining_time / max(1, len(starts) - i), 30)
        
        try:
            opt_points, score = _optimize_from_start(start, max_iter=100, time_budget=time_budget)
            if score > best_score:
                best_score = score
                best_points = opt_points
        except Exception:
            continue
    
    # If no optimization succeeded, use the best initial configuration
    if best_points is None:
        for start in starts:
            s = _score(start)
            if s > best_score:
                best_score = s
                best_points = start
    
    return best_points
# EVOLVE-BLOCK-END
