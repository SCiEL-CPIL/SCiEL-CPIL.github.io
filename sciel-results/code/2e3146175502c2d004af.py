import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import itertools

N = 13

# EVOLVE-BLOCK-START
def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns negative score for minimization.
    Vectorized for speed.
    """
    points = np.asarray(points)
    n = len(points)
    
    # 1. Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0

    # 2. Calculate Minimum Triangle Area (Vectorized)
    # Generate indices for all triples
    # This creates a (286, 3) array of indices
    indices = np.array(list(itertools.combinations(range(n), 3)))
    i, j, k = indices[:, 0], indices[:, 1], indices[:, 2]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    v1 = p2 - p1
    v2 = p3 - p1
    
    # Cross product z-component: v1x*v2y - v1y*v2x
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area <= 1e-12:
        return 0.0
        
    score = min_area / hull_area
    return -score

def construct():
    # Define bounds for differential evolution
    # We can restrict the search space to a square [-1, 1] x [-1, 1]
    # The hull area will vary, but the ratio is invariant to scaling/rotation/translation.
    # However, DE works better with bounded variables.
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Objective function wrapper
    def objective(x):
        points = x.reshape(-1, 2)
        return _score(points)
    
    # Run Differential Evolution
    # popsize=15, maxiter=100 is a good balance for 26 dimensions within time limits.
    # We use 'best' strategy for updating.
    try:
        res_de = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            popsize=15, 
            maxiter=100, 
            tol=1e-5,
            mutation=(0.5, 1.5),
            recombination=0.9,
            polish=False # We will polish manually to save time if needed, or let it do it
        )
    except Exception:
        # Fallback to a simple regular polygon if DE fails
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_x = res_de.x
    
    # Local Refinement with Nelder-Mead to get precise coordinates
    # This helps escape the coarse granularity of DE
    try:
        res_nm = minimize(
            objective, 
            best_x, 
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        best_x = res_nm.x
    except Exception:
        pass # Keep DE result if NM fails
    
    opt_points = best_x.reshape(-1, 2)
    return opt_points
# EVOLVE-BLOCK-END
