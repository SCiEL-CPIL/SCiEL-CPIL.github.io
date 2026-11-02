"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import random


def _triangle_area(p1, p2, p3):
    """Compute the area of a triangle given three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _min_triangle_area(pts):
    """Compute the minimum triangle area over all triples of points."""
    n = len(pts)
    min_area = float('inf')
    for i, j, k in combinations(range(n), 3):
        area = _triangle_area(pts[i], pts[j], pts[k])
        if area < min_area:
            min_area = area
    return min_area


def _convex_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    
    min_area = _min_triangle_area(pts)
    return min_area / hull_area


def _objective(params):
    """Objective function to minimize (negative of the score)."""
    pts = params.reshape(-1, 2)
    score = _score_points(pts)
    return -score


def construct():
    """Construct 13 points to maximize the minimum triangle area ratio."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Start with a regular polygon as initial guess
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_params = initial.flatten()
    best_score = _score_points(initial)
    
    # Use differential evolution for global optimization
    bounds = [(-1, 1)] * (2 * N)
    
    # Run differential evolution
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            x0=initial.flatten(),
            maxiter=100,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False,
            callback=lambda x, convergence: None
        )
        
        if result.fun < 0:  # Only if score is positive
            candidate = result.x.reshape(-1, 2)
            candidate_score = _score_points(candidate)
            if candidate_score > best_score:
                best_score = candidate_score
                best_params = result.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_params.reshape(-1, 2)
    
    # Local refinement using L-BFGS-B
    try:
        local_result = minimize(
            _objective,
            best_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if local_result.fun < 0:
            candidate = local_result.x.reshape(-1, 2)
            candidate_score = _score_points(candidate)
            if candidate_score > best_score:
                best_score = candidate_score
                best_params = local_result.x
    except Exception:
        pass
    
    return best_params.reshape(-1, 2)
# EVOLVE-BLOCK-END
