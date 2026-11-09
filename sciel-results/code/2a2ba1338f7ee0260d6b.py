# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution
import time


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y) coordinates of the 14 points.

    """

    n = 14
    d = 3

    start_time = time.time()
    deadline = start_time + 500  # Leave 100s margin

    # Objective function to minimize (negative of the metric)
    def objective(x):
        if time.time() > deadline:
            return np.inf
        
        pts = x.reshape((n, d))
        
        # Center the points to remove translation invariance
        pts = pts - np.mean(pts, axis=0)
        
        # Calculate pairwise distances
        dists = pdist(pts)
        
        if len(dists) == 0:
            return np.inf
            
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0:
            return np.inf
            
        ratio = d_min / d_max
        return -ratio**2

    # Initial guess: random points
    np.random.seed(42)
    x0 = np.random.randn(n * d)
    
    # Bounds for differential evolution
    bounds = [(-10, 10)] * (n * d)
    
    # Run differential evolution
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=50, 
            tol=1e-8, 
            popsize=15,
            seed=42,
            polish=True
        )
        best_points = result.x.reshape((n, d))
    except Exception as e:
        # Fallback to random if optimization fails
        best_points = x0.reshape((n, d))

    # Normalize: center and scale
    best_points = best_points - np.mean(best_points, axis=0)
    norm = np.linalg.norm(best_points, axis=1)
    if np.max(norm) > 0:
        best_points = best_points / np.max(norm) * 100

    return best_points


# EVOLVE-BLOCK-END
