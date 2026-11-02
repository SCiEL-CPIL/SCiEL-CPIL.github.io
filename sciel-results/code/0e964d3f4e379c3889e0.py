"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
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

def _soft_min(arr, T):
    """
    Soft minimum approximation using log-sum-exp.
    """
    min_val = np.min(arr)
    exp_terms = np.exp(-T * (arr - min_val))
    log_sum_exp = np.log(np.sum(exp_terms))
    return min_val - (1.0 / T) * log_sum_exp

def _gaps_to_points(gaps):
    """
    Convert 13 gaps to 14 points on the unit circle.
    """
    gap_14 = 2 * np.pi - np.sum(gaps)
    all_gaps = np.append(gaps, gap_14)
    angles = np.zeros(N)
    for i in range(1, N):
        angles[i] = angles[i-1] + all_gaps[i-1]
    angles = angles % (2 * np.pi)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _angle_objective(gaps, T=50.0):
    """
    Objective function for angle-based optimization.
    Minimizes - (SoftMinArea / HullArea).
    """
    if np.any(gaps <= 0):
        return 1e10
    
    gap_14 = 2 * np.pi - np.sum(gaps)
    if gap_14 <= 0:
        return 1e10
        
    points = _gaps_to_points(gaps)
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 1e10
        
    if hull_area < 1e-10:
        return 1e10
        
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
            tri_areas = 0.5 * np.abs(cross)
            areas.extend(tri_areas)
            
    areas = np.array(areas)
    
    soft_min_area = _soft_min(areas, T)
    score = soft_min_area / hull_area
    
    return -score

def _cart_obj(points_flat):
    """
    Objective function for Cartesian optimization.
    """
    points = points_flat.reshape(N, 2)
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
    """
    start_time = time.time()
    deadline = start_time + 130.0
    
    # 1. Initialization: Regular 14-gon
    regular_gap = 2 * np.pi / N
    initial_gaps = np.full(N - 1, regular_gap)
    
    best_gaps = initial_gaps.copy()
    best_score_val = 0.0
    
    init_points = _gaps_to_points(best_gaps)
    best_score_val = _score(init_points)
    
    # 2. Optimize gaps using L-BFGS-B
    lower_bound = 1e-4
    upper_bound = 2 * np.pi - 13 * lower_bound
    bounds = [(lower_bound, upper_bound)] * (N - 1)
    
    if time.time() < deadline - 40:
        try:
            res = minimize(
                _angle_objective,
                best_gaps,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.fun < -best_score_val:
                new_gaps = res.x
                gap_14 = 2 * np.pi - np.sum(new_gaps)
                if np.all(new_gaps > 0) and gap_14 > 0:
                    new_points = _gaps_to_points(new_gaps)
                    new_score = _score(new_points)
                    if new_score > best_score_val:
                        best_score_val = new_score
                        best_gaps = new_gaps
        except Exception:
            pass
            
    # 3. Nelder-Mead on gaps to polish
    if time.time() < deadline - 20:
        try:
            res_nm = minimize(
                _angle_objective,
                best_gaps,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < -best_score_val: # Note: res.fun is -score
                new_gaps = res_nm.x
                gap_14 = 2 * np.pi - np.sum(new_gaps)
                if np.all(new_gaps > 0) and gap_14 > 0:
                    new_points = _gaps_to_points(new_gaps)
                    new_score = _score(new_points)
                    if new_score > best_score_val:
                        best_score_val = new_score
                        best_gaps = new_gaps
        except Exception:
            pass
            
    # 4. Final Polish in Cartesian Space
    final_points = _gaps_to_points(best_gaps)
    
    if time.time() < deadline - 5:
        try:
            res_cart = minimize(
                _cart_obj,
                final_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            current_score = -res_cart.fun
            if current_score > best_score_val:
                best_score_val = current_score
                final_points = res_cart.x.reshape(N, 2)
        except Exception:
            pass
            
    # 5. Normalization
    try:
        hull = ConvexHull(final_points)
        hull_area = hull.volume
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
        
    return final_points
# EVOLVE-BLOCK-END
