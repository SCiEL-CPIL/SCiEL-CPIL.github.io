"""
Strategy: Full 28-dimensional optimization with robust multi-start local search.
Abandons the central symmetry constraint to allow for asymmetric optimal configurations.
Uses a fast vectorized scoring function and Powell's method for efficient local refinement.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    Vectorized implementation.
    """
    n = len(points)
    if n < 3:
        return float('inf')
    
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    for i in range(n - 2):
        # Vectors from i to j and k
        # j > i, k > j
        # dx_ij = x[j] - x[i], dy_ij = y[j] - y[i]
        # dx_ik = x[k] - x[i], dy_ik = y[k] - y[i]
        # area = 0.5 * |dx_ij * dy_ik - dx_ik * dy_ij|
        
        # Vectorize over j and k for fixed i?
        # It's easier to just loop i, and vectorize j,k pairs.
        # But the number of pairs is small enough (14*13/2 = 91 pairs, then k>j).
        # Total triples C(14,3) = 364.
        
        # Let's use a slightly more vectorized approach for fixed i
        # Indices j = i+1 to n-1
        # Indices k = j+1 to n-1
        
        # We can create arrays of x[j], y[j] and x[k], y[k]
        # But k depends on j.
        
        # Alternative: compute all areas for fixed i
        # dx_j = x[i+1:] - x[i]
        # dy_j = y[i+1:] - y[i]
        # For each j_idx (corresponding to j = i+1+j_idx), k ranges from j+1 to n-1
        # So k_idx ranges from j_idx+1 to n-1-(i+1) = n-i-2
        
        dx_j = x[i+1:] - x[i]
        dy_j = y[i+1:] - y[i]
        
        # For each j, we need cross product with all k > j
        # Let's just loop j inside the i loop, it's fast enough.
        for j_idx in range(len(dx_j)):
            j = i + 1 + j_idx
            dx_ij = dx_j[j_idx]
            dy_ij = dy_j[j_idx]
            
            # k > j
            if j + 1 >= n:
                continue
            
            dx_ik = x[j+1:] - x[i]
            dy_ik = y[j+1:] - y[i]
            
            cross = dx_ij * dy_ik - dx_ik * dy_ij
            areas = 0.5 * np.abs(cross)
            
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
                    if min_area < 1e-10:
                        return min_area
                        
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

def _objective(params):
    """
    Objective function to minimize (negative score).
    params: flat array of 28 coordinates.
    """
    points = params.reshape(N, 2)
    
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
        
    score = _score(points)
    
    if score < 1e-6:
        return 1e10
        
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_params = init_points.flatten().copy()
    best_score_val = -_objective(best_params)
    
    # Multi-start local search with Powell's method
    # Powell is derivative-free and robust for this type of problem.
    
    # Define a few initial perturbations
    rng = np.random.default_rng(42)
    
    num_starts = 10
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        # Generate initial point
        if i == 0:
            # Start with regular polygon
            p_init = init_points.copy()
        else:
            # Perturb the regular polygon
            # Add some noise to the radii and angles
            noise_scale = 0.2 * (i / num_starts) # Increase noise with start index? Or random.
            # Let's just use random points within a disk
            # Or better, perturb the regular polygon
            p_init = init_points.copy()
            # Add random noise
            noise = rng.normal(0, 0.3, size=(N, 2))
            p_init = p_init + noise
            
        # Optimize
        try:
            res = minimize(
                _objective,
                p_init.flatten(),
                method='Powell',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if res.fun < _objective(best_params):
                best_params = res.x
                best_score_val = -res.fun
        except Exception:
            pass

    # Additional refinement with Nelder-Mead on the best result
    if time.time() < deadline - 10:
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

    # Reconstruct final points
    points = best_params.reshape(N, 2)
    
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
