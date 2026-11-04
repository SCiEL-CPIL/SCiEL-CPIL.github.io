import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 14

def _score_points(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    try:
        # Compute Convex Hull Area
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
        
        if hull_area == 0:
            return 0.0

        # Compute Min Triangle Area
        # Vectorized calculation
        # i, j, k
        min_area = float('inf')
        
        # Pre-compute indices for speed? 
        # 14 points -> 364 triangles. 
        # Looping in Python is slow if done many times. 
        # But we can vectorize the triangle area calculation.
        
        # p_i, p_j, p_k
        # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
        
        # Create index arrays
        # This is done inside the function, which is called many times.
        # Better to precompute indices globally.
        
        global TRIPLE_INDICES
        if 'TRIPLE_INDICES' not in globals() or TRIPLE_INDICES is None:
             TRIPLE_INDICES = np.array(list(itertools.combinations(range(N), 3)))
             
        i_idx = TRIPLE_INDICES[:, 0]
        j_idx = TRIPLE_INDICES[:, 1]
        k_idx = TRIPLE_INDICES[:, 2]
        
        p_i = points[i_idx]
        p_j = points[j_idx]
        p_k = points[k_idx]
        
        # Cross product (2D)
        # (pj - pi) x (pk - pi)
        # (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
        
        dx_j = p_j[:, 0] - p_i[:, 0]
        dy_j = p_j[:, 1] - p_i[:, 1]
        dx_k = p_k[:, 0] - p_i[:, 0]
        dy_k = p_k[:, 1] - p_i[:, 1]
        
        cross = dx_j * dy_k - dy_j * dx_k
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        
        return min_area / hull_area
        
    except Exception:
        return 0.0

# Precompute indices
TRIPLE_INDICES = np.array(list(itertools.combinations(range(N), 3)))

def _make_ring_points(n_out, n_in, r, theta):
    """
    Generate points for a two-ring configuration.
    Outer ring: n_out points, radius 1, angles 0..2pi.
    Inner ring: n_in points, radius r, angles theta..theta+2pi.
    """
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    points = np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])
    return points

def _neg_score_params(params, n_out, n_in):
    r, theta = params
    if r <= 0 or r >= 1:
        return 0.0
    points = _make_ring_points(n_out, n_in, r, theta)
    return -_score_points(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_points = None
    best_score = 0.0
    
    # Try different splits
    splits = [(5, 9), (6, 8), (7, 7)]
    
    for n_out, n_in in splits:
        if time.time() > deadline - 30: # Leave time for polish
            break
            
        bounds = [(0.1, 0.9), (0, 2*np.pi/n_in)]
        
        # Use Differential Evolution for global search
        try:
            res = differential_evolution(
                _neg_score_params, 
                bounds, 
                args=(n_out, n_in),
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-5,
                polish=False # We will do a separate polish
            )
            
            if res.success:
                r_opt, theta_opt = res.x
                score = -res.fun
                points = _make_ring_points(n_out, n_in, r_opt, theta_opt)
                
                if score > best_score:
                    best_score = score
                    best_points = points
                    
        except Exception as e:
            pass
            
        # Local refinement on parameters
        if best_points is not None and n_out == best_points.shape[0] // 2: # Rough check, better to store params
             pass # We don't store params separately, just points. 
             # Let's just run minimize on the points found by DE for this split if it was good.
             
    # If we have a good candidate, do coordinate-level polish
    if best_points is not None:
        if time.time() < deadline - 20:
            try:
                res_polish = minimize(
                    lambda p: -_score_points(p.reshape(N, 2)),
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 2000}
                )
                
                polished_points = res_polish.x.reshape(N, 2)
                polished_score = -res_polish.fun
                
                if polished_score > best_score:
                    best_score = polished_score
                    best_points = polished_points
                    
            except Exception as e:
                pass

    if best_points is None:
        # Fallback: Regular 14-gon? Or just the 7/7 ring with default params
        best_points = _make_ring_points(7, 7, 0.5, 0)
        
    return best_points
