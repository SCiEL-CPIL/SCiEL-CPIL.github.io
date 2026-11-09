import numpy as np
from scipy.optimize import differential_evolution

def construct():
    n = 8
    
    def objective(x):
        pts = x.reshape(n, 2)
        # Calculate pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        valid_dists = dists[mask]
        
        # If any distance is 0, return large penalty
        if np.any(valid_dists < 1e-12):
            return -1e10
            
        # Compute diameter
        diam = np.max(valid_dists)
        if diam < 1e-12:
            return -1e10
            
        # Scale to diameter 2
        scale = 2.0 / diam
        scaled_dists = valid_dists * scale
        
        # The score is proportional to the product of distances.
        # Maximizing the product is equivalent to maximizing the sum of logs.
        # The actual score is Delta / n^n, but since n is constant, we maximize Delta.
        # Delta = prod(dists)
        # log(Delta) = sum(log(dists))
        return np.sum(np.log(scaled_dists))

    # Define the search space: each coordinate in [-1, 1]
    # This ensures we cover the unit square. The optimizer will find the best configuration.
    # We use a wide enough box to allow the shape to form.
    bounds = [(-1.0, 1.0)] * (n * 2)
    
    # Run Differential Evolution
    # This is a global optimizer, so it handles the non-convexity well.
    # maxiter and popsize control the runtime. 120s limit is generous for n=8.
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=1000, 
            popsize=20,
            tol=1e-8,
            init='sobol'
        )
        if res.success:
            best_pts = res.x.reshape(n, 2)
        else:
            # Fallback to regular octagon if DE fails
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            best_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    except Exception:
        # Fallback to regular octagon on error
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_pts = np.column_stack((np.cos(angles), np.sin(angles)))
        
    # Scale the final points to diameter 2 to satisfy the output contract
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    valid_dists = dists[mask]
    current_diam = np.max(valid_dists)
    
    if current_diam > 1e-12:
        scale_factor = 2.0 / current_diam
        final_pts = best_pts * scale_factor
    else:
        # Degenerate case, fallback to regular octagon
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        final_pts = np.column_stack((np.cos(angles), np.sin(angles)))

    return final_pts.tolist()
