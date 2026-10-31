"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull


def _triangle_areas(points):
    """Compute all triangle areas for n points."""
    n = len(points)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                areas[i, j, k] = 0.5 * abs(cross)
    return areas


def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
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
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: compute area using shoelace formula on sorted points
        angles = np.arctan2(points[:, 1] - points[:, 1].mean(), 
                            points[:, 0] - points[:, 0].mean())
        sorted_points = points[np.argsort(angles)]
        n = len(sorted_points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += sorted_points[i, 0] * sorted_points[j, 1]
            area -= sorted_points[j, 0] * sorted_points[i, 1]
        return abs(area) / 2.0


def _score(points):
    """Compute score: min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area


def _soft_min_triangle_area(points, beta=100.0):
    """Smooth surrogate for min triangle area using log-sum-exp."""
    n = len(points)
    # Compute all triangle areas
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                areas.append(0.5 * abs(cross))
    
    if not areas:
        return 0.0
    
    areas = np.array(areas)
    # Soft min: -log-sum-exp(-areas/beta) * beta
    # For numerical stability, subtract max
    max_area = np.max(areas)
    log_sum_exp = np.log(np.sum(np.exp(-(areas - max_area) / beta)))
    soft_min = max_area - beta * log_sum_exp
    return soft_min


def _objective_for_de(x):
    """Objective function for differential evolution (to minimize negative score)."""
    points = x.reshape(-1, 2)
    score = _score(points)
    return -score


def _local_refine(points, deadline):
    """Local refinement using gradient-based optimization."""
    current_points = points.copy()
    current_score = _score(current_points)
    
    def neg_score(x):
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    # Try multiple local optimizations from different starting points
    best_points = current_points
    best_score = current_score
    
    for _ in range(5):
        if time.time() > deadline:
            break
        
        # Add small perturbation
        perturbed = current_points + np.random.randn(*current_points.shape) * 0.01
        result = minimize(neg_score, perturbed.flatten(), method='L-BFGS-B', 
                         options={'maxiter': 100, 'ftol': 1e-10})
        
        if result.success or result.fun < -best_score:
            candidate = result.x.reshape(-1, 2)
            candidate_score = _score(candidate)
            if candidate_score > best_score:
                best_points = candidate
                best_score = candidate_score
                current_points = candidate
    
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Initial configuration: points on a circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radius = 1.0
    initial_points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    
    # Also try a grid-like configuration
    grid_points = np.array([[i % 4, i // 4] for i in range(N)])
    grid_points = (grid_points - grid_points.mean(axis=0)) / (grid_points.std(axis=0) + 1e-10)
    
    best_points = initial_points
    best_score = _score(initial_points)
    
    # Try grid configuration
    grid_score = _score(grid_points)
    if grid_score > best_score:
        best_points = grid_points
        best_score = grid_score
    
    # Differential evolution
    bounds = [(-2, 2)] * (2 * N)
    if time.time() < deadline - 60:
        result = differential_evolution(
            _objective_for_de, 
            bounds, 
            seed=42, 
            maxiter=50, 
            tol=1e-8,
            polish=False,
            updating='immediate'
        )
        
        if result.fun < -best_score:
            de_points = result.x.reshape(-1, 2)
            de_score = _score(de_points)
            if de_score > best_score:
                best_points = de_points
                best_score = de_score
    
    # Local refinement
    if time.time() < deadline - 10:
        refined_points, refined_score = _local_refine(best_points, deadline)
        if refined_score > best_score:
            best_points = refined_points
            best_score = refined_score
    
    return best_points
# EVOLVE-BLOCK-END
