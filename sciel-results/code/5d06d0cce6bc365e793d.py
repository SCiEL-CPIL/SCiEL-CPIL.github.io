"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    Vectorized implementation for speed.
    """
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    # Loop over i and j, vectorize over k
    for i in range(N):
        for j in range(i + 1, N):
            if j + 1 >= N:
                continue
            
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            
            # k ranges from j+1 to N-1
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
                    
    return min_area

def _score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-10:
        return 0.0
        
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _points_from_params(params):
    """
    Reconstruct points from angles and radii.
    params: array of shape (2*N,).
    First N are angles, next N are radii.
    """
    angles = params[:N]
    radii = params[N:]
    
    # Ensure radii are positive
    radii = np.abs(radii) + 1e-6
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    return np.column_stack((x, y))

def _objective(params):
    """
    Objective function to minimize (negative score).
    """
    points = _points_from_params(params)
    
    # Check for NaNs/Infs
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
        
    score = _score(points)
    
    if score < 1e-6:
        return 1e10
        
    return -score

def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    
    Strategy:
    1. Parameterize points by angles and radii to enforce convex position.
    2. Initialize with a regular 14-gon.
    3. Use Powell's method (derivative-free, robust for non-smooth min-area)
       to optimize the parameters.
    4. Normalize the best result to unit area before returning.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial guess: Regular 14-gon
    # Angles are evenly spaced
    init_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Radii are all 1.0
    init_radii = np.ones(N)
    
    init_params = np.concatenate((init_angles, init_radii))
    
    best_params = init_params.copy()
    best_score_val = -_objective(best_params)
    
    # Optimization
    # Powell is good for this kind of problem because it doesn't require gradients
    # and can handle the non-smoothness of the min() operation better than gradient methods.
    
    # We might need to restart if Powell gets stuck, but let's try a robust single run first
    # with high maxiter.
    
    try:
        res = minimize(
            _objective,
            best_params,
            method='Powell',
            options={
                'maxiter': 5000,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        
        if time.time() < deadline:
            if res.fun < _objective(best_params):
                best_params = res.x
                best_score_val = -res.fun
                
    except Exception:
        pass

    # If we have time, try a second phase with Nelder-Mead on the best found
    if time.time() < deadline - 10:
        try:
            res2 = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if res2.fun < _objective(best_params):
                best_params = res2.x
                best_score_val = -res2.fun
        except Exception:
            pass

    # Reconstruct final points
    points = _points_from_params(best_params)
    
    # Normalize to unit area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        # Fallback to regular polygon if hull fails
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
        return final_points
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = points * scale
    else:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale

    return final_points
# EVOLVE-BLOCK-END
