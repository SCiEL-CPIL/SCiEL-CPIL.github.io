"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

DEADLINE = 150.0
START_TIME = time.time()
SAFETY_MARGIN = 5.0


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # For 2D, volume is area
    except Exception:
        # Fallback: if points are collinear or degenerate
        return 0.0


def _score(points):
    """Compute the normalized minimum triangle area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _regular_polygon(n, center=(0.5, 0.5), radius=0.5):
    """Generate a regular n-gon centered at center with given radius."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])


def construct():
    t0 = time.time()
    
    # Start with a regular polygon as initial guess
    best_points = _regular_polygon(N)
    best_score = _score(best_points)
    
    # Try optimization from the regular polygon
    def objective(x):
        if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
            return 0.0
        points = x.reshape(-1, 2)
        return -_score(points)  # Minimize negative score
    
    x0 = best_points.flatten()
    
    # Try multiple random perturbations and optimize from each
    rng = np.random.RandomState(42)
    
    for restart in range(20):
        if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
            break
        
        # Perturb the initial point
        if restart == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + rng.normal(0, 0.05, size=x0.shape)
        
        try:
            result = minimize(objective, x_start, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if result.success or result.fun < -best_score:
                candidate = result.x.reshape(-1, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = candidate
        except Exception:
            continue
    
    # Return the best points found
    return best_points
# EVOLVE-BLOCK-END
