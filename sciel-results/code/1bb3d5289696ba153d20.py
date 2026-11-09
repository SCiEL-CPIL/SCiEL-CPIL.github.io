# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    d = 2
    
    def objective(flat_points):
        points = flat_points.reshape(n, 2)
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        ratio = d_min / d_max
        # We want to maximize ratio^2, so minimize negative
        return -(ratio * ratio)

    # Bounds for each coordinate: [-1, 1]
    # This constrains the points to a square. The optimal shape is scale invariant,
    # so finding the best shape in this box is equivalent to finding the global best shape.
    bounds = [(-1.0, 1.0)] * (n * d)
    
    # Differential Evolution is a global optimization algorithm.
    # It uses a population of candidate solutions and evolves them.
    # This is distinct from the local Powell method used in the seed.
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=15, 
            tol=1e-8, 
            polish=True, 
            init='sobol'
        )
        best_points = res.x.reshape(n, 2)
    except Exception:
        # Fallback to a simple grid or circle if DE fails or times out (unlikely in 600s for 32 dims)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
# EVOLVE-BLOCK-END
