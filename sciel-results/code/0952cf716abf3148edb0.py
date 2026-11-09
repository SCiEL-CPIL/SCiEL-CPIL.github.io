# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    bounds = [(-1.0, 1.0)] * (n * d)
    
    def objective(x):
        pts = x.reshape(n, d)
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min < 1e-8:
            return 1e10
        # We want to minimize d_max / d_min
        # This is equivalent to maximizing d_min / d_max
        return d_max / d_min

    # Run Differential Evolution
    # maxiter=1000, popsize=20
    # This is a global optimization approach, distinct from the local search (SA) in the seed.
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=1000, 
            popsize=20, 
            tol=1e-8, 
            seed=42,
            polish=False,
            init='sobol'
        )
        best_pts = result.x.reshape(n, d)
    except Exception:
        # Fallback to random if DE fails
        np.random.seed(42)
        best_pts = np.random.uniform(-1, 1, (n, d))

    return best_pts

# EVOLVE-BLOCK-END
