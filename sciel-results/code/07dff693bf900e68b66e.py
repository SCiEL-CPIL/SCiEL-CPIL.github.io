"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def _score_fast(points):
    """
    Fast calculation of min triangle area / convex hull area.
    points: (N, 2) array.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # 1. Calculate all triangle areas vectorized
    # Points: (N, 2)
    # We need all triples (i, j, k) with i < j < k
    # This is N choose 3. For N=14, 364 triangles.
    # Broadcasting:
    # p_i: (N, 1, 2)
    # p_j: (1, N, 2)
    # p_k: (1, 1, N, 2)
    # This creates (N, N, N, 2) which is 14^3 = 2744 points, then filter.
    # Alternatively, use indices.
    
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    # Filter i < j < k
    mask = (i < j) & (j < k)
    i_f = i[mask]
    j_f = j[mask]
    k_f = k[mask]
    
    p_i = points[i_f] # (M, 2)
    p_j = points[j_f] # (M, 2)
    p_k = points[k_f] # (M, 2)
    
    # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (x_k - x_i)(y_j - y_i) |
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_tri_area = np.min(areas)
    
    # 2. Calculate Convex Hull Area
    # Use scipy.spatial.ConvexHull for robustness, or manual.
    # Manual is faster for small N if optimized, but scipy is safer.
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    return min_tri_area / hull_area

def construct():
    """
    Place 14 points in a planar convex region to maximize the minimum triangle area
    normalized by the convex hull area.
    
    Strategy:
    1. Use Differential Evolution for global search.
    2. Polish with Nelder-Mead.
    """
    
    def objective(x):
        pts = x.reshape(N, 2)
        # Normalize to ensure hull area is not too small or large to help DE?
        # No, the ratio is scale invariant. But DE needs bounds.
        # Bounds [-2, 2] should be enough.
        try:
            s = _score_fast(pts)
            return -s
        except Exception:
            return 1e6 # Large penalty

    # Bounds for each coordinate
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    # Differential Evolution
    # popsize=10, maxiter=50 is a good balance for 120s limit.
    # seed=42 for reproducibility in evaluation? The task says "each evaluation", 
    # but usually we want the best possible.
    # We will run DE once.
    
    start_time = time.time()
    
    try:
        result_de = differential_evolution(objective, bounds, maxiter=100, popsize=15, tol=1e-8, seed=42, polish=False)
        best_x = result_de.x
        best_score = -result_de.fun
    except Exception:
        # Fallback to regular polygon if DE fails
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_x = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
        best_score = _score_fast(best_x.reshape(N, 2))
        
    # Check time
    if time.time() - start_time > 100:
        return best_x.reshape(N, 2)
        
    # Polish with Nelder-Mead
    try:
        result_nm = minimize(objective, best_x, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        if result_nm.fun < best_score: # Remember objective is negative
            best_x = result_nm.x
            best_score = -result_nm.fun
    except Exception:
        pass
        
    return best_x.reshape(N, 2)
# EVOLVE-BLOCK-END
