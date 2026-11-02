"""
Strategy: Centrally symmetric polygon optimization using Differential Evolution + Nelder-Mead.
Exploits the symmetry of optimal Heilbronn configurations for even N.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
M = N // 2  # 7 pairs of antipodal points

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    Vectorized implementation for speed.
    """
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
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

def _points_from_sym_params(params):
    """
    Reconstruct points from 7 independent 2D vectors (centrally symmetric).
    params: array of shape (14,) -> 7 vectors (x1, y1, ..., x7, y7)
    The points are v1, v2, ..., v7, -v1, -v2, ..., -v7.
    """
    vecs = params.reshape(M, 2)
    # To ensure convexity and avoid self-intersection or degenerate hulls,
    # we should ideally sort by angle, but for optimization we can just return
    # the points and let the score function penalize bad configurations.
    # However, for ConvexHull to work correctly and represent the intended polygon,
    # the points should be in order. 
    # A robust way: compute angles and sort.
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    # Sort the vectors by angle
    sorted_indices = np.argsort(angles)
    sorted_vecs = vecs[sorted_indices]
    
    # Create full set of points: sorted_vecs and -sorted_vecs (reversed to maintain convex order?)
    # Actually, if we take sorted_vecs and append -sorted_vecs[::-1], we get a convex polygon
    # if the vectors are in angular order around the origin.
    # Let's verify: v1 (angle a1), v2 (a2)... v7 (a7).
    # -v1 has angle a1 + pi.
    # If we list v1..v7, then -v7..-v1, the angles are a1..a7, a7+pi..a1+pi.
    # This forms a closed loop around the origin.
    pts_pos = sorted_vecs
    pts_neg = -sorted_vecs[::-1]
    points = np.vstack((pts_pos, pts_neg))
    
    return points

def _objective(params):
    """
    Objective function to minimize (negative score).
    Uses a soft minimum to improve differentiability for local optimizers.
    """
    points = _points_from_sym_params(params)
    
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
        
    score = _score(points)
    
    if score < 1e-6:
        return 1e10
        
    return -score

def _objective_soft(params):
    """
    Objective using a soft-min for the triangle areas to smooth the landscape.
    """
    points = _points_from_sym_params(params)
    
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
        
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 1e10
        
    if hull_area < 1e-10:
        return 1e10
        
    # Compute all triangle areas
    x = points[:, 0]
    y = points[:, 1]
    areas = []
    for i in range(N):
        for j in range(i + 1, N):
            if j + 1 >= N:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            a = 0.5 * np.abs(cross)
            areas.extend(a)
            
    areas = np.array(areas)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    beta = 50.0
    # Shift for numerical stability
    shifted = areas - np.min(areas)
    soft_min_val = -1.0/beta * np.log(np.sum(np.exp(-beta * shifted))) + np.min(areas)
    
    score = soft_min_val / hull_area
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial guess: Regular 14-gon
    # Vectors for regular 14-gon: angles k*pi/7 for k=0..6
    init_angles = np.linspace(0, np.pi, M, endpoint=False)
    init_radii = np.ones(M)
    init_vecs = np.column_stack((init_radii * np.cos(init_angles), init_radii * np.sin(init_angles)))
    init_params = init_vecs.flatten()
    
    best_params = init_params.copy()
    best_score_val = -_objective(best_params)
    
    # Phase 1: Differential Evolution for global exploration
    # Bounds: Radii between 0.1 and 2.0, Angles between 0 and 2*pi?
    # Actually, since we sort by angle, the specific angle range doesn't matter as much,
    # but the relative angles do.
    # Let's bound the coordinates of the vectors.
    # A regular 14-gon has radius 1. Let's allow radius 0.5 to 1.5.
    # So x, y in [-1.5, 1.5]
    bounds = [(-1.5, 1.5)] * (2 * M)
    
    if time.time() < deadline - 60:
        try:
            # Use DE with the hard objective for robustness
            de_res = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-6,
                popsize=15,
                polish=True,  # DE internally uses Nelder-Mead for polish
                init='latinhypercube'
            )
            
            if de_res.fun < _objective(best_params):
                best_params = de_res.x
                best_score_val = -de_res.fun
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead on the best found from DE
    if time.time() < deadline - 20:
        try:
            nm_res = minimize(
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
            if nm_res.fun < _objective(best_params):
                best_params = nm_res.x
                best_score_val = -nm_res.fun
        except Exception:
            pass
            
    # Phase 3: Soft objective polish to navigate flat regions or sharp minima
    if time.time() < deadline - 10:
        try:
            nm_soft_res = minimize(
                _objective_soft,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            # Check if the soft result is actually better with the hard metric
            hard_score_soft_params = -_objective(nm_soft_res.x)
            if hard_score_soft_params > best_score_val:
                best_params = nm_soft_res.x
                best_score_val = hard_score_soft_params
        except Exception:
            pass

    # Reconstruct final points
    points = _points_from_sym_params(best_params)
    
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
