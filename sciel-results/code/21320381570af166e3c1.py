# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses an optimization approach to maximize the sum of log distances, which correlates with maximizing the min distance.
    """
    n = 16
    d = 2
    
    # Initial guess: 4x4 grid
    # Center it at 0
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Normalize initial points
    # Fix centroid to 0
    points -= np.mean(points, axis=0)
    # Fix scale: sum of squares = n (so avg sq dist from origin is 1)
    scale = np.sqrt(np.sum(points**2) / n)
    points /= scale

    def objective(params):
        # params is a flat array of size 32
        pts = params.reshape(n, d)
        
        # Enforce constraints softly or hard?
        # Let's use a penalty method or just rely on the optimizer if we don't use constraints.
        # Actually, it's better to use scipy's constraints.
        
        # Calculate pairwise distances
        # d[i, j] is distance between i and j
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        
        # Avoid log(0)
        dists = np.maximum(dists, 1e-9)
        
        # We want to maximize sum of log(dist) for i < j
        # Sum over upper triangle
        iu = np.triu_indices(n, k=1)
        sum_log_dist = np.sum(np.log(dists[iu]))
        
        return -sum_log_dist

    # Constraints
    # 1. Centroid is 0
    # 2. Sum of squares is n (normalization)
    
    def centroid_x(params):
        pts = params.reshape(n, d)
        return np.sum(pts[:, 0])
        
    def centroid_y(params):
        pts = params.reshape(n, d)
        return np.sum(pts[:, 1])
        
    def norm_sq(params):
        pts = params.reshape(n, d)
        return np.sum(pts**2) - n

    constraints = [
        {'type': 'eq', 'fun': centroid_x},
        {'type': 'eq', 'fun': centroid_y},
        {'type': 'eq', 'fun': norm_sq}
    ]
    
    # Initial parameters
    x0 = points.ravel()
    
    # Run optimization
    # Try a few different methods or just a robust one
    # L-BFGS-B is good for large N, but N=16 is small.
    # Nelder-Mead is derivative-free and robust for non-smooth-ish things, but our objective is smooth.
    # Let's use L-BFGS-B with constraints.
    
    try:
        res = minimize(objective, x0, method='SLSQP', constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-9})
        if res.success:
            points = res.x.reshape(n, d)
        else:
            # If it fails, try Nelder-Mead without constraints but with normalization in objective?
            # Or just return the initial grid which is decent.
            pass
    except Exception:
        pass
    
    # Final check: ensure it's finite and shape is correct
    if not np.all(np.isfinite(points)):
        points = np.random.randn(n, d)
        
    # The problem asks to return the points. The metric is scale invariant, so the specific scale doesn't matter, 
    # but the shape must be (16, 2).
    return points

# EVOLVE-BLOCK-END
