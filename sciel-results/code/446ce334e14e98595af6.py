import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import itertools

N = 13

# EVOLVE-BLOCK-START
def _precompute_triples():
    """Precompute indices for all C(N, 3) triples."""
    return np.array(list(itertools.combinations(range(N), 3)))

TRIPLES = _precompute_triples()

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns negative score for minimization.
    Vectorized for speed.
    """
    points = np.asarray(points)
    
    # 1. Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0 # Degenerate hull
    except Exception:
        return 0.0

    # 2. Calculate Minimum Triangle Area (Vectorized)
    # TRIPLES has shape (286, 3)
    # p_i, p_j, p_k
    p1 = points[TRIPLES[:, 0]]
    p2 = points[TRIPLES[:, 1]]
    p3 = points[TRIPLES[:, 2]]
    
    # Cross product magnitude for 2D vectors: x1*y2 - y1*x2
    v1 = p2 - p1
    v2 = p3 - p1
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    min_area = np.min(areas)
    
    if min_area <= 1e-12:
        return 0.0
        
    score = min_area / hull_area
    return -score

def construct():
    # Define bounds for differential evolution
    # The convex hull area will be normalized, so absolute scale doesn't matter,
    # but keeping points in a reasonable box helps DE convergence.
    bounds = [(-1, 1)] * (2 * N)
    
    # 1. Global Search with Differential Evolution
    # DE is stochastic, so set seed for reproducibility if needed, but random is fine.
    # maxiter and popsize are tuned to fit within 120s.
    # 26 parameters. Popsize 10-20 is small but effective for DE.
    res_de = differential_evolution(
        _score, 
        bounds, 
        maxiter=100, 
        popsize=15, 
        tol=1e-4, 
        seed=42,
        polish=False # We will do our own polish
    )
    
    x_best = res_de.x
    
    # 2. Local Refinement with Nelder-Mead
    # Start from the best point found by DE
    def objective(x):
        points = x.reshape(-1, 2)
        return _score(points)
        
    res_nm = minimize(
        objective, 
        x_best, 
        method='Nelder-Mead', 
        options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7}
    )
    
    opt_points = res_nm.x.reshape(-1, 2)
    
    # Optional: Try a few random restarts if time permits? 
    # DE usually finds a good global basin. NM refines it.
    # Given the 120s limit, one DE run + NM is a good balance.
    
    return opt_points
# EVOLVE-BLOCK-END
