import numpy as np
from scipy.optimize import differential_evolution

def construct():
    n_points = 16
    
    def objective(x):
        pts = x.reshape(-1, 2)
        # Calculate all pairwise distances
        # Using broadcasting for efficiency
        # pts shape (16, 2)
        # diff shape (16, 16, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Get upper triangle to avoid zeros on diagonal and duplicates
        iu = np.triu_indices(n_points, 1)
        pairwise_dists = dists[iu]
        
        d_min = np.min(pairwise_dists)
        d_max = np.max(pairwise_dists)
        
        if d_min < 1e-8:
            return 1e10
            
        return (d_max / d_min) ** 2

    # Bounds for each coordinate
    # Since the problem is scale invariant, we can pick any reasonable bound.
    # If the optimal shape is 'small', it will fit. If 'large', it will hit bounds.
    # A box of [-10, 10] is generous enough for typical unit-scaled optimal configs.
    bounds = [(-10, 10)] * (n_points * 2)
    
    # Run differential evolution
    # seed=0 for reproducibility, though task allows any valid result
    # maxiter=100, popsize=15 is a reasonable balance for 32 dims
    # tol=1e-4 to stop early if converged
    result = differential_evolution(objective, bounds, maxiter=200, popsize=20, seed=42, tol=1e-6, polish=True)
    
    # The optimal points
    optimal_pts = result.x.reshape(-1, 2)
    
    return optimal_pts
