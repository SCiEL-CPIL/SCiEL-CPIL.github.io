"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time

N = 14

# Precompute indices for all triples (i, j, k) with i < j < k
def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                indices.append((i, j, k))
    return np.array(indices).T

TRIPLE_INDICES = _get_triple_indices(N)
I_IDX = TRIPLE_INDICES[0]
J_IDX = TRIPLE_INDICES[1]
K_IDX = TRIPLE_INDICES[2]

def _fast_score(points):
    """
    points: shape (N, 2)
    returns: min_triangle_area / convex_hull_area
    """
    pi = points[I_IDX]
    pj = points[J_IDX]
    pk = points[K_IDX]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_area = np.min(areas)
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area <= 1e-10:
        return 0.0
        
    return min_area / hull_area

def _objective(flat_points):
    """
    Objective for minimization: negative of the score.
    flat_points: shape (2*N,)
    """
    points = flat_points.reshape(N, 2)
    score = _fast_score(points)
    return -score

def construct():
    """
    Use Differential Evolution for global search and Nelder-Mead for local polishing
    to maximize the minimum triangle area relative to the convex hull area.
    """
    start_time = time.time()
    time_limit = 110.0
    
    bounds = [(0.0, 1.0) for _ in range(2 * N)]
    
    best_result = None
    best_score = -np.inf
    
    # 1. Global Search with Differential Evolution
    try:
        de_result = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            popsize=20, 
            maxiter=200, 
            tol=1e-6,
            init='sobol'
        )
        
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_result = de_result.x.copy()
            
    except Exception:
        pass
    
    # 2. Local Polishing with Nelder-Mead
    if best_result is not None:
        elapsed = time.time() - start_time
        if elapsed < time_limit - 5:
            try:
                nm_result = minimize(
                    _objective, 
                    best_result, 
                    method='Nelder-Mead', 
                    options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                
                if nm_result.fun < best_score:
                    best_score = nm_result.fun
                    best_result = nm_result.x.copy()
            except Exception:
                pass
    
    # Fallback if DE failed
    if best_result is None:
        for _ in range(20):
            if time.time() - start_time > time_limit - 5:
                break
            init = np.random.uniform(0, 1, 2 * N)
            try:
                nm_res = minimize(_objective, init, method='Nelder-Mead', options={'maxiter': 1000})
                if nm_res.fun < best_score:
                    best_score = nm_res.fun
                    best_result = nm_res.x.copy()
            except Exception:
                continue

    if best_result is None:
        best_result = np.random.uniform(0, 1, 2 * N)
        
    return best_result.reshape(N, 2)

# EVOLVE-BLOCK-END
