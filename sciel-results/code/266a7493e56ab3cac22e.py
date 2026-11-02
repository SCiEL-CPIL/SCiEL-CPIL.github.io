"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

DEADLINE_MARGIN = 10.0  # seconds before wall-clock limit to stop


def _triangle_areas(points):
    """Compute areas of all triangles formed by triples of points."""
    n = len(points)
    areas = np.empty((n, n, n))
    areas[:] = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1])
                              - (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                areas[i, j, k] = a
                areas[j, i, k] = a
                areas[k, i, j] = a
    return areas


def _min_triangle_area(points):
    """Minimum area over all triples."""
    n = len(points)
    min_a = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1])
                              - (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if a < min_a:
                    min_a = a
    return min_a


def _hull_area(points):
    """Area of convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    ha = _hull_area(points)
    if ha <= 1e-12:
        return 0.0
    return _min_triangle_area(points) / ha


def _soft_min(values, temperature=0.05):
    """Soft-min surrogate for smooth optimization."""
    return -temperature * np.log(np.sum(np.exp(-values / temperature)))


def _objective(params, temperature=0.05):
    """
    Objective to minimize: negative soft-min of triangle areas / hull area.
    params: 2*N array of point coordinates.
    """
    points = params.reshape(-1, 2)
    ha = _hull_area(points)
    if ha <= 1e-12:
        return 1e6
    # Compute all triangle areas
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1])
                              - (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                areas.append(a)
    areas = np.array(areas)
    sm = _soft_min(areas, temperature)
    return -(sm / ha)


def _refine(points, start_time, max_iter=200):
    """Refine points using L-BFGS-B with adaptive temperature."""
    best_points = points.copy()
    best_score = _score(points)
    
    # Start with a higher temperature and decrease
    temperatures = [0.1, 0.05, 0.02, 0.01, 0.005]
    
    for temp in temperatures:
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        params = best_points.flatten()
        
        def obj(p):
            return _objective(p, temperature=temp)
        
        try:
            res = minimize(obj, params, method='L-BFGS-B', options={'maxiter': max_iter, 'gtol': 1e-8})
            new_points = res.x.reshape(-1, 2)
            new_score = _score(new_points)
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
        except Exception:
            pass
    
    return best_points, best_score


def construct():
    start_time = time.time()
    
    # Initial guesses: regular polygon, circle, and random
    best_points = None
    best_score = -np.inf
    
    def try_points(pts):
        nonlocal best_points, best_score
        if time.time() - start_time > DEADLINE_MARGIN:
            return
        s = _score(pts)
        if s > best_score:
            best_points = pts.copy()
            best_score = s
    
    # Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    try_points(pts)
    
    # Slightly perturbed regular polygons
    for seed in range(20):
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        rng = np.random.RandomState(seed)
        perturb = rng.randn(N, 2) * 0.1
        pts = np.column_stack([np.cos(angles) + perturb[:, 0], np.sin(angles) + perturb[:, 1]])
        try_points(pts)
    
    # Random points in unit square
    for seed in range(30):
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        rng = np.random.RandomState(seed + 100)
        pts = rng.rand(N, 2)
        try_points(pts)
    
    # Random points on a circle with varying radii
    for seed in range(20):
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        rng = np.random.RandomState(seed + 200)
        angles = np.sort(rng.rand(N) * 2 * np.pi)
        radii = 1 + rng.randn(N) * 0.2
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        try_points(pts)
    
    if best_points is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Refine the best point
    if time.time() - start_time < DEADLINE_MARGIN:
        best_points, best_score = _refine(best_points, start_time)
    
    # Try refining from a few other good initializations
    for seed in range(5):
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        rng = np.random.RandomState(seed + 500)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.randn(N) * 0.05
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        refined_pts, refined_score = _refine(pts, start_time)
        if refined_score > best_score:
            best_points = refined_pts
            best_score = refined_score
    
    return best_points
# EVOLVE-BLOCK-END
