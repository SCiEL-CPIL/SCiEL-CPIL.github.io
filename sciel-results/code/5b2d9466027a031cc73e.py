"""Rewritten program for Heilbronn convex regions n=14."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time
import itertools

N = 14

# EVOLVE-BLOCK-START

def _triangle_areas(points):
    """Compute areas of all triangles formed by triples of points.
    Returns an array of shape (C(N,3),) with doubled areas (2*area)."""
    n = len(points)
    idx = np.arange(n)
    # Use vectorized approach: for each pair (i,j), cross product with k
    # More efficient: iterate over i < j < k
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(pj - pi, pk - pi)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                areas.append(abs(cross) * 0.5)
    return np.array(areas)


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: if all points are collinear or fewer than 3
        return 1e-10


def _score(points):
    """Compute the score: minimum triangle area / convex hull area."""
    tri_areas = _triangle_areas(points)
    min_tri = np.min(tri_areas)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    return min_tri / hull_area


def _soft_min(values, temperature=0.01):
    """Soft minimum approximation using log-sum-exp trick."""
    # -log-sum-exp(-x/T) / T approximates min(x)
    scaled = -values / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -temperature * log_sum


def _objective_softmin(points, temperature=0.01):
    """Objective to minimize: negative soft-min of triangle areas, normalized by hull area.
    We want to maximize min(tri_areas) / hull_area.
    So minimize -min(tri_areas)/hull_area.
    
    Using soft-min for differentiability."""
    tri_areas = _triangle_areas(points)
    # Normalize by hull area to make scale-invariant
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    normalized = tri_areas / hull_area
    soft_min_val = _soft_min(normalized, temperature)
    return -soft_min_val


def _initial_convex_polygon(n, rng):
    """Generate initial points as a convex polygon with random angles and radii."""
    angles = np.sort(rng.random(n) * 2 * np.pi)
    # Ensure roughly uniform spacing but with some randomness
    radii = 0.5 + 0.5 * rng.random(n)
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    # Center at origin
    points -= np.mean(points, axis=0)
    return points


def construct():
    """Construct 14 points to maximize minimum triangle area / convex hull area."""
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s kill
    
    rng = np.random.RandomState(N)
    
    best_points = None
    best_score = -1.0
    
    # Multiple restarts with different initializations
    n_restarts = 20
    max_iter_per_restart = 2000
    
    for restart in range(n_restarts):
        if time.time() - start_time > deadline - 10.0:
            break
        
        # Generate initial convex polygon
        points = _initial_convex_polygon(N, rng)
        
        # Flatten for scipy optimizer
        x0 = points.flatten()
        
        # Define objective function for scipy (minimize)
        def obj(x):
            pts = x.reshape(-1, 2)
            return _objective_softmin(pts, temperature=0.05)
        
        # Try to optimize
        try:
            result = minimize(
                obj, x0,
                method='BFGS',
                options={'maxiter': max_iter_per_restart, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < obj(x0):
                opt_points = result.x.reshape(-1, 2)
                score = _score(opt_points)
                
                if score > best_score:
                    best_score = score
                    best_points = opt_points.copy()
                    
        except Exception:
            continue
    
    # If no valid result, return a default convex polygon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points

# EVOLVE-BLOCK-END
