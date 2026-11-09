import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools
import sys

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area_fast(points):
    """
    Fast calculation of minimum triangle area using vectorized operations.
    Points: (N, 2) array. Returns minimum area (not doubled).
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            v1 = points[j] - pi
            v2s = points[j + 1:] - pi
            cross_vals = v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0]
            areas = 0.5 * np.abs(cross_vals)
            if len(areas) > 0:
                cur_min = np.min(areas)
                if cur_min < min_area:
                    min_area = cur_min
    return min_area

def _score(points):
    """
    Score = Min Triangle Area / Convex Hull Area.
    """
    try:
        if np.allclose(points, points[0]):
            return 0.0
        
        hull = ConvexHull(points)
        hull_area = hull.volume
        
        if hull_area < 1e-12:
            return 0.0
            
        min_tri_area = _min_triangle_area_fast(points)
        
        if min_tri_area < 1e-12:
            return 0.0
            
        return min_tri_area / hull_area
    except Exception:
        return 0.0

def _negative_score(params):
    """
    Objective function for minimization.
    params: array of shape (2*N,), where params[0::2] are radii and params[1::2] are angles.
    Returns negative score.
    """
    radii = params[0::2]
    angles = params[1::2]
    
    # Ensure radii are positive and within a reasonable bound
    radii = np.clip(radii, 0.1, 10.0)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack([x, y])
    
    score = _score(points)
    return -score

def _regular_polygon_params(n_points=N):
    """
    Initialize parameters for a regular polygon.
    """
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    radii = np.ones(n_points)
    params = np.zeros(2 * n_points)
    params[0::2] = radii
    params[1::2] = angles
    return params

def _jittered_polygon_params(n_points=N, seed=42, jitter=0.1):
    """
    Initialize parameters for a jittered regular polygon.
    """
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False) + rng.normal(0, jitter, n_points)
    radii = np.ones(n_points) + rng.normal(0, 0.05, n_points)
    radii = np.clip(radii, 0.5, 2.0)
    
    params = np.zeros(2 * n_points)
    params[0::2] = radii
    params[1::2] = angles
    return params

def _random_params(n_points=N, seed=42):
    """
    Initialize random parameters.
    """
    rng = np.random.default_rng(seed)
    radii = rng.uniform(0.5, 1.5, n_points)
    angles = rng.uniform(0, 2 * np.pi, n_points)
    
    params = np.zeros(2 * n_points)
    params[0::2] = radii
    params[1::2] = angles
    return params

def _params_to_points(params):
    """
    Convert parameters to points.
    """
    radii = params[0::2]
    angles = params[1::2]
    radii = np.clip(radii, 0.1, 10.0)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = 0.0
    
    def update_best(points, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_points = points.copy()
            best_score = score
            
    # Strategy: Use Powell optimization on radial-angular parameters.
    # This allows points to move off the convex hull boundary, which might improve the score.
    
    initializations = []
    
    # 1. Regular polygon
    initializations.append(_regular_polygon_params())
    
    # 2. Jittered regular polygons
    for seed in [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]:
        initializations.append(_jittered_polygon_params(seed=seed, jitter=0.1))
        
    # 3. Random initializations
    for seed in [42, 123, 456, 789, 1011]:
        initializations.append(_random_params(seed=seed))
        
    # First pass: Quick evaluation of initializations
    for init in initializations:
        if time.time() > deadline - 10:
            break
        points = _params_to_points(init)
        score = _score(points)
        update_best(points, score)
        
    # Second pass: Powell optimization from top initializations
    scored_inits = []
    for init in initializations:
        points = _params_to_points(init)
        score = _score(points)
        scored_inits.append((score, init))
        
    scored_inits.sort(key=lambda x: -x[0])
    
    top_k = min(5, len(scored_inits))
    for i, (score, init_params) in enumerate(scored_inits[:top_k]):
        if time.time() > deadline - 20:
            break
            
        try:
            result = minimize(
                _negative_score,
                init_params,
                method='Powell',
                options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            opt_params = result.x
            opt_points = _params_to_points(opt_params)
            opt_score = _score(opt_points)
            update_best(opt_points, opt_score)
        except Exception:
            pass
            
    # Third pass: Refine the best solution with a more detailed Powell run
    if best_points is not None and time.time() < deadline - 30:
        # Extract parameters from best points
        x, y = best_points[:, 0], best_points[:, 1]
        radii = np.sqrt(x**2 + y**2)
        angles = np.arctan2(y, x)
        
        best_params = np.zeros(2 * N)
        best_params[0::2] = radii
        best_params[1::2] = angles
        
        try:
            result = minimize(
                _negative_score,
                best_params,
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7}
            )
            opt_params = result.x
            opt_points = _params_to_points(opt_params)
            opt_score = _score(opt_points)
            update_best(opt_points, opt_score)
        except Exception:
            pass
            
    if best_points is None:
        # Fallback: regular polygon
        best_points = _params_to_points(_regular_polygon_params())
        
    return best_points

# EVOLVE-BLOCK-END
