from scipy.optimize import minimize
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar

N = 14

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    # Vectorized calculation
    n = len(points)
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    
    p_i = points[i_idx[mask]]
    p_j = points[j_idx[mask]]
    p_k = points[k_idx[mask]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    # Cross product in 2D
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _get_ring_points(r_in, r_out, theta_offset):
    """
    Generate points for two concentric regular heptagons.
    """
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_outer = np.linspace(theta_offset, 2*np.pi + theta_offset, 7, endpoint=False)
    
    inner_pts = np.column_stack([r_in * np.cos(angles_inner), r_in * np.sin(angles_inner)])
    outer_pts = np.column_stack([r_out * np.cos(angles_outer), r_out * np.sin(angles_outer)])
    
    return np.vstack([inner_pts, outer_pts])

def _score_ring_params(params):
    """
    Score function for ring parameters: [r_in, r_out, theta_offset]
    """
    r_in, r_out, theta_offset = params
    if r_in < 0 or r_out <= 0 or r_in >= r_out:
        return 0.0
        
    points = _get_ring_points(r_in, r_out, theta_offset)
    return _score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin for 150s limit
    
    # Strategy: Two concentric regular heptagons.
    # Variables: r_in, r_out, theta_offset
    # We can fix r_out = 1.0 due to scale invariance (ratio is scale invariant).
    # So variables are: r_in, theta_offset.
    
    # Initial best from simple heuristics
    best_score = 0
    best_params = None
    
    # 1. Grid search for coarse initialization
    r_in_vals = np.linspace(0.1, 0.9, 10)
    theta_vals = np.linspace(0, np.pi/7, 10)
    
    for r_in in r_in_vals:
        for theta in theta_vals:
            s = _score_ring_params([r_in, 1.0, theta])
            if s > best_score:
                best_score = s
                best_params = [r_in, 1.0, theta]
    
    if best_params is None:
        # Fallback: Regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
        
    # 2. Local optimization using Nelder-Mead or BFGS on the 3 parameters (r_in, r_out, theta)
    # We allow r_out to vary slightly to ensure we find the global scale if needed, 
    # though it should be 1.
    
    def _neg_score_params(params):
        return -_score_ring_params(params)
    
    # Use multiple restarts around the best found point
    current_best_params = best_params
    current_best_score = best_score
    
    # Optimization 1: From the grid best
    if time.time() < deadline - 10:
        try:
            res = minimize(_neg_score_params, current_best_params, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < -current_best_score:
                current_best_score = -res.fun
                current_best_params = res.x
        except Exception:
            pass

    # Optimization 2: Try some other symmetric configurations
    # e.g. theta = 0 (aligned), theta = pi/7 (interleaved)
    for theta_init in [0, np.pi/7]:
        for r_in_init in [0.3, 0.5, 0.7]:
            if time.time() < deadline - 10:
                try:
                    p_init = [r_in_init, 1.0, theta_init]
                    res = minimize(_neg_score_params, p_init, method='Nelder-Mead',
                                   options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
                    if res.fun < -current_best_score:
                        current_best_score = -res.fun
                        current_best_params = res.x
                except Exception:
                    pass

    # If we have time, do a finer grid search to ensure we didn't miss a local optimum
    # This is a structural change: relying on the 3-parameter family which is much smaller than 28-dim
    
    # Final check and return
    final_points = _get_ring_points(current_best_params[0], current_best_params[1], current_best_params[2])
    final_score = _score(final_points)
    
    # Just in case, compare with a regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    reg14 = np.column_stack([np.cos(angles), np.sin(angles)])
    reg14_score = _score(reg14)
    
    if reg14_score > final_score:
        return reg14
        
    return final_points
