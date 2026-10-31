"""Heilbronn convex regions: place 13 points to maximize min triangle area / hull area."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 13
DEADLINE = 115.0  # seconds, safety margin before 120s kill

def _compute_min_triangle_area(points):
    """Compute minimum triangle area among all triples."""
    n = points.shape[0]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _compute_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # For 2D, volume is area
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / hull area."""
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri = _compute_min_triangle_area(points)
    return min_tri / hull_area

def _objective(params):
    """Objective function for optimization (minimize negative score)."""
    points = params.reshape(N, 2)
    s = _score(points)
    return -s

def _points_from_params(params):
    """
    Convert parameters to points.
    params: array of length 2*N
    First N are radii, next N are angular shifts.
    Base angles are equally spaced.
    """
    radii = params[:N]
    angles_shift = params[N:]
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = base_angles + angles_shift
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

def construct():
    start_time = time.time()
    
    # Initial guess: regular 13-gon with radius 1.0
    # params: [r1...r13, a1...a13]
    # Start with all radii 1.0 and shifts 0.0
    init_params = np.zeros(2 * N)
    init_params[:N] = 1.0
    
    best_score = 0.0
    best_points = _points_from_params(init_params)
    best_params = init_params.copy()
    
    # 1. Global search with Differential Evolution
    # Bounds: radii in [0.1, 2.0], angles in [-0.5, 0.5]
    bounds = [(0.1, 2.0)] * N + [(-0.5, 0.5)] * N
    
    if time.time() - start_time < DEADLINE - 30:
        try:
            # Use differential evolution for global search
            de_res = differential_evolution(
                _objective, 
                bounds, 
                seed=42, 
                maxiter=100, 
                popsize=20,
                tol=0.01
            )
            if de_res.fun < -best_score:
                best_score = -de_res.fun
                best_params = de_res.x
                best_points = _points_from_params(best_params)
        except Exception:
            pass
    
    # 2. Local refinement with Nelder-Mead
    if time.time() - start_time < DEADLINE - 10:
        try:
            nm_res = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if nm_res.fun < -best_score:
                best_score = -nm_res.fun
                best_params = nm_res.x
                best_points = _points_from_params(best_params)
        except Exception:
            pass
            
    return best_points
