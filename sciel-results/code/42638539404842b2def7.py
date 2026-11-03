"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

def _score(points):
    """
    Compute min triangle area / convex hull area.
    Returns negative score for minimization.
    """
    points = np.asarray(points, dtype=np.float64).reshape(N, 2)
    
    # Check for degenerate hull
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 1.0 # Return high cost (low score) if invalid
        
    if hull_area <= 1e-10:
        return 1.0
        
    min_area = float('inf')
    for i in range(N):
        p1 = points[i]
        for j in range(i + 1, N):
            p2 = points[j]
            v1 = p2 - p1
            for k in range(j + 1, N):
                p3 = points[k]
                v2 = p3 - p1
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
                    if min_area < 1e-10:
                        return 1.0
                        
    return - (min_area / hull_area)

def construct():
    """
    Uses Differential Evolution followed by Nelder-Mead refinement.
    """
    # Define bounds: each point coordinate in [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    best_points = None
    best_score_val = -1.0
    
    # 1. Global Search with Differential Evolution
    try:
        result = differential_evolution(
            _score,
            bounds,
            popsize=15,
            maxiter=100,
            tol=1e-6,
            seed=42,
            workers=1,
            init='latinhypercube' # Better initial spread
        )
        if result.x is not None:
            best_points = result.x.reshape(N, 2)
            best_score_val = -result.fun
    except Exception:
        pass

    # Fallback if DE failed or didn't produce a result
    if best_points is None:
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score_val = -_score(best_points)

    # 2. Local Refinement with Nelder-Mead
    # Run multiple times from slightly perturbed starting points?
    # Or just one run from the DE result.
    try:
        res_nm = minimize(
            _score,
            best_points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if res_nm.fun < _score(best_points): # Check if improved
            best_points = res_nm.x.reshape(N, 2)
            best_score_val = -res_nm.fun
    except Exception:
        pass

    return best_points

# EVOLVE-BLOCK-END
