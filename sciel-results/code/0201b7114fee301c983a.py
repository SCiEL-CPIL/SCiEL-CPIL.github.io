"""Heilbronn convex regions: place 13 points to maximize min triangle area / hull area."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

N = 13
_DEADLINE = 150.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE - _elapsed()


def compute_min_triangle_area(points):
    """Compute minimum triangle area among all triples of points."""
    n = points.shape[0]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) -
                                 (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area


def compute_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = compute_min_triangle_area(points)
    return min_area / hull_area


def _soft_min(arr, beta):
    """Soft minimum approximation using log-sum-exp."""
    return -np.log(np.sum(np.exp(-beta * arr))) / beta


def _all_triangle_areas(points):
    """Compute all C(n,3) triangle areas as a flat array."""
    n = points.shape[0]
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) -
                                 (p3[0] - p1[0]) * (p2[1] - p1[1]))
                areas.append(area)
    return np.array(areas)


def _objective_soft(x, beta):
    """Objective for soft-min annealing: negative soft-min of triangle areas / hull area."""
    points = x.reshape(-1, 2)
    hull_area = compute_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    areas = _all_triangle_areas(points)
    sm = _soft_min(areas, beta)
    return -sm / hull_area


def _optimize_from_seed(seed_points, beta_schedule, max_iter=200, tol=1e-8):
    """Optimize from a seed using soft-min annealing with L-BFGS-B."""
    best_points = seed_points.copy()
    best_score = score(best_points)
    
    for beta in beta_schedule:
        if _remaining() < 5.0:
            break
        
        def obj(x):
            return _objective_soft(x, beta)
        
        x0 = best_points.flatten()
        try:
            res = minimize(obj, x0, method='L-BFGS-B', maxiter=max_iter, tol=tol)
            if res.fun < 1e9:
                candidate = res.x.reshape(-1, 2)
                s = score(candidate)
                if s > best_score:
                    best_score = s
                    best_points = candidate
        except Exception:
            pass
    
    return best_points, best_score


def _generate_grid_seeds():
    """Generate diverse initial seeds."""
    seeds = []
    rng = np.random.RandomState(42)
    
    # Regular grid-like patterns
    for spacing in [0.3, 0.4, 0.5]:
        pts = []
        for i in range(4):
            for j in range(4):
                if len(pts) < 13:
                    pts.append([i * spacing - 0.75, j * spacing - 0.75])
        if len(pts) == 13:
            seeds.append(np.array(pts))
    
    # Hexagonal packing
    pts = []
    row_h = np.sqrt(3) / 2
    for row in range(4):
        offset = 0 if row % 2 == 0 else 0.5
        for col in range(4):
            if len(pts) < 13:
                pts.append([col + offset, row * row_h])
    if len(pts) == 13:
        # Center and scale
        pts = np.array(pts)
        pts -= pts.mean(axis=0)
        pts /= np.max(np.abs(pts))
        seeds.append(pts)
    
    # Random points in unit square
    for _ in range(5):
        pts = rng.rand(13, 2)
        seeds.append(pts)
    
    # Points on a circle with some interior
    angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    interior = rng.rand(4, 2) * 0.5 + 0.25
    pts = np.vstack([pts, interior])
    seeds.append(pts)
    
    return seeds


def construct():
    """Construct 13 points maximizing min triangle area / hull area."""
    if _remaining() < 10.0:
        # Not enough time, return a reasonable default
        rng = np.random.RandomState(N)
        return rng.rand(N, 2)
    
    seeds = _generate_grid_seeds()
    
    # Beta schedule for annealing: start coarse, end fine
    beta_schedule = [5, 10, 20, 50, 100, 200, 500, 1000]
    
    best_points = seeds[0]
    best_score = score(best_points)
    
    for seed in seeds:
        if _remaining() < 15.0:
            break
        
        # Try optimization from this seed
        pts, s = _optimize_from_seed(seed, beta_schedule, max_iter=100)
        if s > best_score:
            best_score = s
            best_points = pts
    
    # Final polish with higher beta
    if _remaining() > 20.0:
        final_beta = [2000, 5000]
        pts, s = _optimize_from_seed(best_points, final_beta, max_iter=150)
        if s > best_score:
            best_score = s
            best_points = pts
    
    # Ensure points are well-formed
    if best_points.shape != (N, 2):
        rng = np.random.RandomState(N)
        best_points = rng.rand(N, 2)
    
    return best_points
