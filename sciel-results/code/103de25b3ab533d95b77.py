import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time

N = 14
N_OUT = 7
N_IN = 7

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _generate_points(params):
    """
    Generate 14 points from parameters.
    params: [r_out, r_in, theta_0, ..., theta_13]
    """
    r_out = params[0]
    r_in = params[1]
    angles = params[2:16]
    
    # 7 outer points
    angles_out = angles[0:7]
    x_out = r_out * np.cos(angles_out)
    y_out = r_out * np.sin(angles_out)
    
    # 7 inner points
    angles_in = angles[7:14]
    x_in = r_in * np.cos(angles_in)
    y_in = r_in * np.sin(angles_in)
    
    points = np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])
    return points

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
        
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    min_area = 0.5 * np.min(np.abs(cross))
    
    return min_area / hull_area

def _score_params(params):
    """
    Objective function for optimization.
    """
    try:
        points = _generate_points(params)
        return _calculate_score(points)
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Parameters: [r_out, r_in, 14 angles]
    bounds = [(0.5, 1.0), (0.0, 0.8)] + [(0.0, 2 * np.pi)] * 14
    
    best_score = 0.0
    best_params = None
    
    # 1. Global Search with Differential Evolution
    try:
        de_result = differential_evolution(
            lambda p: -_score_params(p),
            bounds=bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=0.01,
            polish=False,
            init='latinhypercube'
        )
        
        if de_result.fun < 0:
            best_score = -de_result.fun
            best_params = de_result.x
        else:
            best_params = [0.9, 0.5] + list(np.linspace(0, 2*np.pi, 7, endpoint=False)) + list(np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7)
            best_score = _score_params(best_params)
            
    except Exception:
        best_params = [0.9, 0.5] + list(np.linspace(0, 2*np.pi, 7, endpoint=False)) + list(np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7)
        best_score = _score_params(best_params)
        
    # 2. Local Polish with Nelder-Mead
    if time.time() < deadline - 10:
        try:
            res = minimize(
                lambda p: -_score_params(p),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 2000}
            )
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass

    # 3. Final Polish
    if time.time() < deadline - 5:
        try:
            res2 = minimize(
                lambda p: -_score_params(p),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000}
            )
            if -res2.fun > best_score:
                best_score = -res2.fun
                best_params = res2.x
        except Exception:
            pass

    points = _generate_points(best_params)
    
    if len(points) != N:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return points
