import numpy as np
from scipy.optimize import differential_evolution

def construct():
    n = 16
    # We fix P0 at (0,0) and P1 at (1,0) to remove translation and rotation invariance
    # This reduces the search space to 2 * (n - 2) = 28 dimensions
    dim = 2 * (n - 2)
    
    def objective(x):
        # Reshape x to (n-2, 2)
        pts_var = x.reshape((n - 2, 2))
        
        # Construct full points array
        pts = np.zeros((n, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = pts_var
        
        # Calculate pairwise distances
        # Using broadcasting
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Get unique pairwise distances (upper triangle, excluding diagonal)
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        
        min_d = np.min(d)
        if min_d < 1e-8:
            return 1e10
        
        max_d = np.max(d)
        ratio = max_d / min_d
        return ratio**2

    # Bounds for the variable points
    # The optimal configuration is likely within [-5, 5] relative to the fixed points
    bounds = [(-5.0, 5.0)] * dim
    
    # Run Differential Evolution
    # popsize=10, maxiter=500 -> ~5000 evaluations
    result = differential_evolution(
        objective, 
        bounds, 
        maxiter=500, 
        popsize=10, 
        tol=1e-4, 
        init='latinhypercube',
        seed=42,
        polish=True
    )
    
    # Reconstruct the points
    pts_var = result.x.reshape((n - 2, 2))
    pts = np.zeros((n, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = pts_var
    
    return pts
