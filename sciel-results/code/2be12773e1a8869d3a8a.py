import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 13

def _score(points):
    """
    Vectorized calculation of the minimum triangle area divided by convex hull area.
    """
    # 1. Compute all triangle areas
    # Points shape: (13, 2)
    # Indices for all combinations of 3
    i, j, k = np.indices((N, N, N))
    # Filter i < j < k
    mask = (i < j) & (j < k)
    ii, jj, kk = i[mask], j[mask], k[mask]
    
    p_i = points[ii]
    p_j = points[jj]
    p_k = points[kk]
    
    # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    
    min_area = np.min(areas)
    
    # 2. Compute Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area <= 0:
        return 0.0
        
    return min_area / hull_area

def _negative_score(points_flat):
    """
    Objective function for minimizers.
    """
    points = points_flat.reshape(N, 2)
    return -_score(points)

def _generate_symmetric_config(k_inner, r_inner, phi, r_outer=1.0):
    """
    Generate points for a 2-shell symmetric configuration.
    k_inner points on a regular polygon of radius r_inner, rotated by phi.
    (N - k_inner) points on a regular polygon of radius r_outer.
    """
    n_outer = N - k_inner
    points = []
    
    # Inner polygon
    angles_in = np.linspace(0, 2 * np.pi, k_inner, endpoint=False) + phi
    for a in angles_in:
        points.append([r_inner * np.cos(a), r_inner * np.sin(a)])
        
    # Outer polygon
    angles_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    for a in angles_out:
        points.append([r_outer * np.cos(a), r_outer * np.sin(a)])
        
    return np.array(points)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # List of (k_inner, initial_r_inner, initial_phi)
    # We try various symmetric structures
    configs = [
        (0, 0.0, 0.0),       # Regular 13-gon
        (1, 0.0, 0.0),       # Center + 12-gon
        (1, 0.5, 0.0),       # Center-ish + 12-gon
        (2, 0.5, 0.0),       # 2 inner + 11 outer
        (3, 0.5, 0.0),       # 3 inner + 10 outer
        (4, 0.5, 0.0),       # 4 inner + 9 outer
        (5, 0.5, 0.0),       # 5 inner + 8 outer
        (6, 0.5, 0.0),       # 6 inner + 7 outer
    ]
    
    for k_in, r0, phi0 in configs:
        if time.time() > deadline:
            break
            
        # Optimization variables: r_inner, phi
        # We fix r_outer to 1.0 because the score is scale invariant.
        # The shape is determined by r_inner and phi.
        
        def objective(params):
            r_in = params[0]
            phi = params[1]
            pts = _generate_symmetric_config(k_in, r_in, phi)
            return -_score(pts)
        
        x0 = np.array([r0, phi0])
        
        try:
            # Use Nelder-Mead for this low-dimensional (2D) symmetric search
            # It handles the non-smoothness of min() better than gradient methods
            result = minimize(objective, x0, method='Nelder-Mead', 
                              options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            
            if result.success:
                r_best, phi_best = result.x
                pts = _generate_symmetric_config(k_in, r_best, phi_best)
                s = _score(pts)
                
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
        except Exception:
            pass

    # If we found a good symmetric configuration, refine it in full 26D space
    if best_points is not None and time.time() < deadline - 10:
        x0 = best_points.flatten()
        try:
            result = minimize(_negative_score, x0, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if result.fun < -best_score:
                new_pts = result.x.reshape(N, 2)
                new_s = _score(new_pts)
                if new_s > best_score:
                    best_score = new_s
                    best_points = new_pts.copy()
        except Exception:
            pass
            
    # Fallback: If nothing worked, return regular 13-gon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
