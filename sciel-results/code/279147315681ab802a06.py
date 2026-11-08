import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def _symmetric_points(params):
    """
    Generate points from symmetric shell parameters.
    params: [r1, theta1, n1, r2, theta2, n2, ...]
    We assume a partition of N points into shells.
    For N=13, we try (7, 6) or (5, 8) etc.
    Let's fix a partition to keep parameter count low.
    Partition: 7 outer, 6 inner.
    params: [r_out, theta_out, r_in, theta_in]
    """
    n_outer = 7
    n_inner = 6
    
    r_out = params[0]
    theta_out = params[1]
    r_in = params[2]
    theta_in = params[3]
    
    # Outer shell: 7 points
    angles_out = theta_out + 2 * np.pi * np.arange(n_outer) / n_outer
    pts_out = np.column_stack((r_out * np.cos(angles_out), r_out * np.sin(angles_out)))
    
    # Inner shell: 6 points
    angles_in = theta_in + 2 * np.pi * np.arange(n_inner) / n_inner
    pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
    
    return np.vstack([pts_out, pts_in])

def _symmetric_objective(params):
    """
    Objective function for symmetric parametrization.
    Minimizes negative score.
    """
    pts = _symmetric_points(params)
    score = _exact_score(pts)
    return -score

def _all_triangle_areas(points):
    """Vectorized computation of all C(n,3) triangle areas."""
    n = len(points)
    if n < 3:
        return np.array([0.0])
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(points):
    """Compute area of the convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0

def _exact_score(points):
    """Calculate the exact score: min triangle area / hull area."""
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _soft_objective(params, beta):
    """
    Negative soft-min of triangle areas divided by hull area.
    We minimize this to maximize the score.
    """
    points = params.reshape(-1, 2)
    areas = _all_triangle_areas(points)
    
    # Soft min approximation
    # To prevent overflow/underflow in exp
    scaled_areas = -beta * areas
    max_val = np.max(scaled_areas)
    if max_val == -np.inf:
        return 1e10
    log_sum_exp = max_val + np.log(np.sum(np.exp(scaled_areas - max_val)))
    soft_min = -log_sum_exp / beta
    
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 1e10
        
    return -soft_min / hull_area

def construct():
    n = N
    t0 = time.time()
    deadline = 145.0 # Safety margin from 150s limit
    
    best_points = None
    best_score = -np.inf
    
    # 1. Symmetric Search Strategy
    # Parameters: [r_out, theta_out, r_in, theta_in]
    # Bounds: radii > 0, angles [0, 2pi)
    bounds = [(0.1, 2.0), (0, 2 * np.pi), (0.0, 1.5), (0, 2 * np.pi)]
    
    # Initial guess for symmetric optimization
    # Outer radius ~1, inner radius ~0.5
    x0_sym = np.array([1.0, 0.0, 0.5, 0.0])
    
    try:
        # Use differential evolution or Nelder-Mead on the reduced 4D space
        # Nelder-Mead is often faster for low dimensions
        res_sym = minimize(_symmetric_objective, x0_sym, method='Nelder-Mead',
                           options={'maxiter': 1000, 'fatol': 1e-10, 'xatol': 1e-10, 'adaptive': True})
        
        best_sym_params = res_sym.x
        sym_pts = _symmetric_points(best_sym_params)
        sym_score = _exact_score(sym_pts)
        
        if sym_score > best_score:
            best_score = sym_score
            best_points = sym_pts
    except Exception as e:
        pass

    # 2. Local Refinement (Asymmetric)
    # Take the best symmetric point set and refine individual coordinates
    if best_points is not None and time.time() - t0 < deadline - 20:
        try:
            # Nelder-Mead on full 26D space, starting from symmetric config
            # Use a tighter tolerance since we are polishing
            res_local = minimize(lambda x: -_exact_score(x.reshape(-1, 2)), best_points.flatten(), 
                                 method='Nelder-Mead',
                                 options={'maxiter': 500, 'fatol': 1e-12, 'xatol': 1e-12, 'adaptive': True})
            
            local_pts = res_local.x.reshape(-1, 2)
            local_score = _exact_score(local_pts)
            
            if local_score > best_score:
                best_score = local_score
                best_points = local_pts
        except Exception:
            pass

    # Fallback if symmetric search failed or timed out early
    if best_points is None:
        # Return a regular 13-gon as a safe fallback
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
