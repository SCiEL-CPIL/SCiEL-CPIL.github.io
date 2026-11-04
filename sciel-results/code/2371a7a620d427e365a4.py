"""Heilbronn convex regions: place 13 points to maximize min triangle area / hull area."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
import itertools

N = 13

def _compute_score(points):
    """
    Compute the score: minimum triangle area divided by convex hull area.
    Returns 0.0 for invalid configurations.
    """
    points = np.asarray(points)
    if len(points) < 3:
        return 0.0
    
    # Check for duplicate or near-duplicate points
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if np.linalg.norm(points[i] - points[j]) < 1e-10:
                return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area < 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    # Compute minimum triangle area
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area of triangle using cross product
                area = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
                if area < min_area:
                    min_area = area
    
    if min_area < 1e-12:
        return 0.0
    
    return min_area / hull_area

def _negative_score(params):
    """
    Objective function for minimization.
    params: array of shape (N, 2)
    """
    points = params.reshape(N, 2)
    score = _compute_score(points)
    return -score

def construct():
    """
    Construct 13 points in the unit square to maximize the minimum triangle area
    divided by the convex hull area.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin for 150s limit
    
    best_score = 0.0
    best_points = None
    
    # Initial guess: regular 13-gon scaled to fit in unit square
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radius = 0.5
    cx, cy = 0.5, 0.5
    init_points = np.column_stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles)
    ])
    init_params = init_points.flatten()
    
    initial_score = _compute_score(init_points)
    if initial_score > best_score:
        best_score = initial_score
        best_points = init_points.copy()
    
    # Differential evolution over the 26-dimensional space of point coordinates
    # Bounds: each coordinate in [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Use multiple restarts with different seeds if time permits
    seeds = [42, 123, 456, 789, 1011]
    
    for seed in seeds:
        if time.time() > deadline - 20.0:
            break
        
        try:
            result = differential_evolution(
                _negative_score,
                bounds,
                seed=seed,
                maxiter=300,
                popsize=15,
                tol=1e-6,
                init='sobol',
                polish=True
            )
            
            if result.success or result.fun < 0:
                candidate_points = result.x.reshape(N, 2)
                candidate_score = _compute_score(candidate_points)
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points.copy()
        except Exception:
            continue
    
    # Final local polish using Nelder-Mead if time permits
    if best_points is not None and time.time() < deadline - 10.0:
        try:
            from scipy.optimize import minimize
            result = minimize(
                _negative_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            candidate_points = result.x.reshape(N, 2)
            candidate_score = _compute_score(candidate_points)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: return the initial regular polygon
        return init_points
    
    return best_points

# EVOLVE-BLOCK-START
# (The construct function above is the entry point)
# EVOLVE-BLOCK-END
