import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time

N = 14

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    # Vectorized approach might be overkill for N=14, but let's use efficient loops
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            pj = points[j]
            v1 = pj - pi
            for k in range(j + 1, n):
                pk = points[k]
                v2 = pk - pi
                # Cross product z-component
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Compute the area of the convex hull of the points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area. Higher is better."""
    min_tri = _min_triangle_area(points)
    hull_area = _hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    return min_tri / hull_area

def _neg_score(params):
    """Objective function for minimization."""
    points = params.reshape(-1, 2)
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    init_params = init_points.flatten()
    
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    best_params = init_params.copy()
    best_score = _score(init_points)
    
    # 1. Differential Evolution
    # We run DE for a fixed number of iterations.
    # If time allows, we run it multiple times with different seeds.
    num_de_runs = 3
    de_time_budget = 30.0
    
    for i in range(num_de_runs):
        if time.time() > deadline - de_time_budget:
            break
            
        seed_val = 42 + i
        try:
            result_de = differential_evolution(
                _neg_score,
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-5,
                seed=seed_val,
                polish=False
            )
            if result_de.fun < -best_score:
                best_score = -result_de.fun
                best_params = result_de.x
        except Exception:
            pass
    
    # 2. Local Refinement
    # Use Nelder-Mead for non-smooth optimization
    if time.time() < deadline - 10:
        try:
            result_local = minimize(
                _neg_score,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if result_local.fun < -best_score:
                best_score = -result_local.fun
                best_params = result_local.x
        except Exception:
            pass
    
    # 3. Final Check
    final_points = best_params.reshape(-1, 2)
    return final_points
