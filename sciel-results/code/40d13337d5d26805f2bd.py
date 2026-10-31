"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _ratio(points):
    """Compute the score: min triangle area / convex hull area."""
    ma = _min_triangle_area(points)
    ha = _convex_hull_area(points)
    if ha < 1e-10:
        return 0.0
    return ma / ha

def _neg_ratio_flat(x):
    """Objective for minimizer: negative of the ratio."""
    points = x.reshape(N, 2)
    return -_ratio(points)

def construct():
    """
    Construct 13 points to maximize min triangle area / convex hull area.
    Approach: Multi-start Nelder-Mead optimization.
    """
    best_score = 0.0
    best_points = None
    
    # Initial guesses
    # 1. Regular polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    p0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # 2. Slightly perturbed regular polygon
    np.random.seed(42) # Ensure reproducibility for the seed program behavior
    noise = np.random.normal(0, 0.1, (N, 2))
    p1 = p0 + noise
    
    # 3. Random points in unit square (normalized later, but shape matters)
    p2 = np.random.rand(N, 2) * 2 - 1
    
    # 4. Another random set
    p3 = np.random.rand(N, 2) * 2 - 1
    
    initial_points = [p0, p1, p2, p3]
    
    # Run Nelder-Mead for each initial point
    start_time = time.time()
    timeout = 110.0  # Leave some buffer for evaluation overhead
    
    for p_init in initial_points:
        if time.time() - start_time > timeout:
            break
            
        # Flatten for scipy
        x0 = p_init.flatten()
        
        # Nelder-Mead options
        try:
            res = minimize(
                _neg_ratio_flat,
                x0,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'maxiter': 5000,
                    'maxfev': 5000
                }
            )
        except Exception:
            continue
        
        if time.time() - start_time > timeout:
            break
            
        score = -res.fun
        if score > best_score:
            best_score = score
            best_points = res.x.reshape(N, 2)
            
    # If we didn't find anything or best is bad, fallback to initial
    if best_points is None:
        best_points = p0
    else:
        # Final polish with a tighter Nelder-Mead or Powell
        try:
            res_final = minimize(
                _neg_ratio_flat,
                best_points.flatten(),
                method='Powell',
                options={
                    'maxiter': 1000,
                    'xtol': 1e-10,
                    'ftol': 1e-10
                }
            )
            score_final = -res_final.fun
            if score_final > best_score:
                best_points = res_final.x.reshape(N, 2)
                best_score = score_final
        except Exception:
            pass

    # Normalize to unit area convex hull
    hull_area = _convex_hull_area(best_points)
    if hull_area > 1e-10:
        scale = 1.0 / np.sqrt(hull_area)
        best_points = best_points * scale
        
    return best_points
# EVOLVE-BLOCK-END
