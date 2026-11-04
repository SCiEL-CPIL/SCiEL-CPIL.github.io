"""Heilbronn triangles: 11 points in unit equilateral triangle, maximize min triangle area."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import math

N = 11
H = math.sqrt(3) / 2

def _tri_area_vec(pts):
    """
    Vectorized minimum triangle area calculation.
    pts: (N, 2) array
    Returns: min area
    """
    # Generate all pairs of points
    # We need to calculate area for all triplets (i, j, k) with i < j < k
    # Using broadcasting to create arrays of indices
    idx = np.arange(N)
    # This creates a meshgrid of indices, but we only need i<j<k
    # A more efficient way for N=11 is to just compute all 165 areas using vectorized operations on subsets
    # Or simply use the loop but with numpy ops inside? No, pure numpy is better.
    
    # Let's use the cross product formula: 0.5 * | (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i) |
    # We can create arrays of all i, j, k combinations.
    i_idx, j_idx, k_idx = np.indices((N, N, N))
    # Mask for i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    
    # Extract coordinates
    pi = pts[i_idx[mask]]
    pj = pts[j_idx[mask]]
    pk = pts[k_idx[mask]]
    
    # Calculate areas
    # Area = 0.5 * abs( (pj_x - pi_x)*(pk_y - pi_y) - (pj_y - pi_y)*(pk_x - pi_x) )
    dx_j = pj[:, 0] - pi[:, 0]
    dy_j = pj[:, 1] - pi[:, 1]
    dx_k = pk[:, 0] - pi[:, 0]
    dy_k = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(dx_j * dy_k - dy_j * dx_k)
    
    return np.min(areas)

def _min_area(pts):
    """
    Wrapper for compatibility, but uses vectorized version.
    """
    if len(pts) < 3:
        return 0.0
    return _tri_area_vec(np.array(pts))

def _inside(p, m=1e-9):
    x, y = p
    if y < -m or x < -m or x > 1 + m:
        return False
    if y > math.sqrt(3) * x + m:
        return False
    if y > math.sqrt(3) * (1 - x) + m:
        return False
    return True

def _project(p, m=1e-8):
    x, y = p
    if y < m:
        y = m
    min_x = y / math.sqrt(3)
    if x < min_x:
        x = min_x
    max_x = 1 - y / math.sqrt(3)
    if x > max_x:
        x = max_x
    if x < m:
        x = m
    if x > 1 - m:
        x = 1 - m
    return [x, y]

def _neg_min_area(flat):
    pts = flat.reshape(N, 2)
    # Project points to ensure they are inside
    pts = np.array([_project(p) for p in pts])
    return -_min_area(pts)

def construct():
    h = H
    
    # Better initial guess based on known structures or grid
    # The optimal 11-point configuration is complex. 
    # We will use a symmetric initial guess and rely on optimization.
    init = [
        [0.0, 0.0], [1.0, 0.0], [0.5, h],
        [0.5, 0.0], [0.75, h/2], [0.25, h/2],
        [1/3, h/3], [2/3, h/3], [0.5, 2*h/3],
        [0.5, h/3],
        [0.5, h/6]
    ]
    init_flat = np.array(init).flatten()
    
    bounds = [(0, 1), (0, h)] * N
    
    best_flat = init_flat.copy()
    best_val = _neg_min_area(best_flat)
    
    # Differential Evolution
    # Use a smaller population for speed, but more iterations? 
    # Or use 'sobol' init which is deterministic and good.
    try:
        res_de = differential_evolution(
            _neg_min_area, bounds, seed=42, maxiter=100, popsize=20,
            tol=1e-8, init='sobol', polish=False, updating='deferred'
        )
        if res_de.fun < best_val:
            best_val = res_de.fun
            best_flat = res_de.x
    except Exception:
        pass
    
    # Local refinement with Powell (often better than Nelder-Mead for this)
    try:
        res_pw = minimize(_neg_min_area, best_flat, method='Powell',
                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res_pw.fun < best_val:
            best_val = res_pw.fun
            best_flat = res_pw.x
    except Exception:
        pass
        
    # Nelder-Mead as a second pass
    try:
        res_nm = minimize(_neg_min_area, best_flat, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res_nm.fun < best_val:
            best_val = res_nm.fun
            best_flat = res_nm.x
    except Exception:
        pass
    
    # Multi-start with random restarts
    rng = np.random.RandomState(123)
    for trial in range(15):
        pts = []
        for _ in range(N):
            u, v = rng.random(2)
            if u + v > 1:
                u, v = 1 - u, 1 - v
            p = [u + 0.5 * v, h * v]
            pts.append(_project(p))
        flat0 = np.array(pts).flatten()
        
        # Quick local refinement for each restart
        try:
            res = minimize(_neg_min_area, flat0, method='Powell',
                           options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11})
            if res.fun < best_val:
                best_val = res.fun
                best_flat = res.x
        except Exception:
            pass
    
    points = best_flat.reshape(N, 2)
    points = np.array([_project(p) for p in points])
    return points
