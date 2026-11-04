import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
import itertools

N = 13

# EVOLVE-BLOCK-START
def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns negative score for minimization.
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
    # Get all combinations of indices
    indices = np.array(list(itertools.combinations(range(N), 3)))
    
    # Extract points for these combinations
    # indices shape: (286, 3)
    # points shape: (13, 2)
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    # Calculate cross product magnitude for all triangles
    # v1 = p2 - p1, v2 = p3 - p1
    # area = 0.5 * |v1[0]*v2[1] - v1[1]*v2[0]|
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area <= 1e-12:
        return 0.0
        
    score = min_area / hull_area
    return -score

def construct():
    # We fix point 0 at (0,0) and point 1 at (1,0) to remove translational and rotational invariance.
    # This reduces the number of free parameters from 2*N to 2*N - 4.
    n_params = 2 * (N - 2)
    
    # Bounds for the remaining 11 points.
    # Since the problem is scale invariant, we can choose arbitrary bounds.
    # A box of [-2, 2] is usually sufficient to contain the convex hull of the optimal configuration
    # given that P0 and P1 are at 0 and 1.
    bounds = [(-2.0, 2.0)] * n_params
    
    def objective(params):
        # Reconstruct points
        points = np.zeros((N, 2))
        points[0] = [0.0, 0.0]
        points[1] = [1.0, 0.0]
        # Fill the rest
        points[2:] = params.reshape(-1, 2)
        
        # Check for duplicates to avoid division by zero or degenerate hulls
        # If any two points are too close, return a bad score
        dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
        # Also check distance between P0 and P1 is 1, which is fixed.
        # Check all pairwise distances? That's expensive.
        # Instead, rely on the ConvexHull exception or area check.
        
        return _score(points)
        
    # Use Differential Evolution for global search
    # tol and maxiter are tuned to fit within time limits
    # popsize 15 is default, might increase for better exploration
    res = differential_evolution(objective, bounds, 
                                 method='de', 
                                 popsize=15, 
                                 maxiter=100, 
                                 tol=1e-4, 
                                 seed=42,
                                 polish=True) # polish=True runs Nelder-Mead on the best result
    
    # Extract the optimized parameters
    opt_params = res.x
    opt_points = np.zeros((N, 2))
    opt_points[0] = [0.0, 0.0]
    opt_points[1] = [1.0, 0.0]
    opt_points[2:] = opt_params.reshape(-1, 2)
    
    return opt_points
# EVOLVE-BLOCK-END
