import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14
N_OUT = 7
N_IN = 7

def _generate_points(r, theta):
    """
    Generate 14 points: 7 on outer unit circle, 7 on inner circle of radius r,
    rotated by theta.
    """
    angles_out = np.linspace(0, 2 * np.pi, N_OUT, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    
    angles_in = np.linspace(0, 2 * np.pi, N_IN, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
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
    
    # Precomputed triple indices for speed
    # We can compute this on the fly or cache it. Caching is better.
    # But for simplicity in the function, we can use a global cache.
    if not hasattr(_calculate_score, 'triple_indices'):
        indices = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    indices.append((i, j, k))
        _calculate_score.triple_indices = np.array(indices)
        
    tri_indices = _calculate_score.triple_indices
    
    p_i = points[tri_indices[:, 0]]
    p_j = points[tri_indices[:, 1]]
    p_k = points[tri_indices[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    min_area = 0.5 * np.min(np.abs(cross))
    
    return min_area / hull_area

def _score_params(params):
    """
    Objective function for optimization.
    params: [r, theta]
    """
    r, theta = params
    if r < 0 or r > 1.0:
        return 0.0
        
    points = _generate_points(r, theta)
    return _calculate_score(points)

def _soft_score_params(params, beta=50.0):
    """
    Soft-min surrogate for smoother optimization if needed.
    """
    r, theta = params
    if r < 0 or r > 1.0:
        return 0.0
        
    points = _generate_points(r, theta)
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
        
    if not hasattr(_calculate_score, 'triple_indices'):
        indices = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    indices.append((i, j, k))
        _calculate_score.triple_indices = np.array(indices)
        
    tri_indices = _calculate_score.triple_indices
    
    p_i = points[tri_indices[:, 0]]
    p_j = points[tri_indices[:, 1]]
    p_k = points[tri_indices[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    areas = 0.5 * np.abs(cross)
    
    # Soft-min
    max_area = np.max(areas)
    shifted = -beta * (areas - max_area)
    log_sum = np.log(np.sum(np.exp(shifted))) + beta * max_area
    soft_min = max_area - log_sum / beta
    
    return soft_min / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # 1. Grid Search
    # r in [0.5, 1.0], theta in [0, 2*pi/7]
    # Due to symmetry, theta range is [0, 2*pi/7]
    theta_max = 2 * np.pi / 7
    
    # Initial coarse grid
    r_vals = np.linspace(0.5, 1.0, 50)
    theta_vals = np.linspace(0, theta_max, 50)
    
    best_score = 0.0
    best_params = (0.9, 0.0)
    
    # Evaluate grid
    for r in r_vals:
        if time.time() > deadline - 10:
            break
        for theta in theta_vals:
            score = _score_params([r, theta])
            if score > best_score:
                best_score = score
                best_params = (r, theta)
                
    # 2. Local Refinement around best grid point
    # Use Nelder-Mead for robustness near non-smooth minima
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda p: -_score_params(p),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000}
            )
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
            
    # 3. Further refinement with Soft-Min surrogate for smoother landscape
    if time.time() < deadline - 5:
        try:
            # Use higher beta for closer approximation to true min
            beta = 100.0
            res_soft = minimize(
                lambda p: -_soft_score_params(p, beta),
                best_params,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0), (0.0, theta_max)],
                options={'ftol': 1e-12, 'gtol': 1e-10, 'maxiter': 500}
            )
            # Verify with exact score
            exact_score = _score_params(res_soft.x)
            if exact_score > best_score:
                best_score = exact_score
                best_params = res_soft.x
        except Exception:
            pass
            
    # 4. Final Polish with Exact Score
    if time.time() < deadline - 2:
        try:
            res_final = minimize(
                lambda p: -_score_params(p),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 500}
            )
            if -res_final.fun > best_score:
                best_score = -res_final.fun
                best_params = res_final.x
        except Exception:
            pass

    # Generate final points
    r, theta = best_params
    points = _generate_points(r, theta)
    
    # Fallback if something went wrong
    if len(points) != N:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return points
