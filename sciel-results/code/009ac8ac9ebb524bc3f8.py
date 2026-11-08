import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 13

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return -1e10
        
    if hull_area < 1e-10:
        return -1e10
        
    n = len(points)
    if n < 3:
        return -1e10

    # Vectorized triangle area calculation
    # Create index arrays for all combinations i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    ii = i_idx[mask]
    jj = j_idx[mask]
    kk = k_idx[mask]
    
    p1 = points[ii]
    p2 = points[jj]
    p3 = points[kk]
    
    v1 = p2 - p1
    v2 = p3 - p1
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _objective(x):
    """
    Objective function for minimization (we minimize negative score).
    """
    points = x.reshape(-1, 2)
    # Optional: Center the points to reduce numerical issues
    # points = points - np.mean(points, axis=0)
    score = _score(points)
    return -score

def construct():
    start_time = time.time()
    # Safety margin: 140s out of 150s
    deadline = start_time + 140.0
    
    # Initial guess: Regular 13-gon
    angles = 2 * np.pi * np.arange(N) / N
    x0 = np.column_stack((np.cos(angles), np.sin(angles))).flatten()
    
    best_x = x0
    best_score = _score(x0.reshape(-1, 2))
    
    # Optimization Strategy
    # 1. Nelder-Mead from regular polygon
    # 2. If time permits, try a few random perturbations of the regular polygon
    
    methods = [
        {'method': 'Nelder-Mead', 'options': {'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}},
        {'method': 'Powell', 'options': {'maxiter': 1000, 'xtol': 1e-8, 'ftol': 1e-10}}
    ]
    
    for m in methods:
        if time.time() > deadline - 10: # Stop if less than 10s left
            break
            
        try:
            res = minimize(_objective, best_x, method=m['method'], options=m['options'])
            
            if time.time() > deadline:
                break
                
            if res.x is not None:
                res_score = _score(res.x.reshape(-1, 2))
                if res_score > best_score:
                    best_x = res.x
                    best_score = res_score
        except Exception:
            continue

    return best_x.reshape(-1, 2)
