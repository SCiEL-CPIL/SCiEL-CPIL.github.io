# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    n_points = 14
    dim = 3
    start_time = time.time()
    timeout = 600  # Safety margin for 630s limit

    def objective(x):
        # Check time limit
        if time.time() - start_time > timeout:
            return 0.0
            
        pts = x.reshape(n_points, dim)
        
        # Center the points to remove translational invariance
        pts = pts - np.mean(pts, axis=0)
        
        # Calculate pairwise distances
        dists = pdist(pts)
        
        if len(dists) == 0:
            return 0.0
            
        d_max = np.max(dists)
        
        if d_max < 1e-12:
            return 0.0
            
        d_min = np.min(dists)
        
        # The metric is (d_min / d_max)^2
        return (d_min / d_max) ** 2

    # Define bounds for Differential Evolution
    # Since the objective is scale invariant, the absolute scale doesn't matter,
    # but we need finite bounds for DE.
    # A range of [-10, 10] is usually sufficient to capture the geometry.
    bounds = [(-10.0, 10.0)] * (n_points * dim)

    # Run Differential Evolution
    # popsize=15, maxiter=1000 should be robust.
    # strategy='best1bin' is standard.
    try:
        result = differential_evolution(
            lambda x: -objective(x), # Minimize negative to maximize
            bounds,
            maxiter=1000,
            popsize=20,
            tol=1e-10,
            seed=42,
            polish=False # We will do our own polish
        )
        best_x = result.x
    except Exception:
        # Fallback to random if DE fails
        best_x = np.random.uniform(-1, 1, n_points * dim)

    # Local Polish with Nelder-Mead
    # Nelder-Mead is derivative-free and handles non-smooth functions well
    try:
        polish_result = minimize(
            lambda x: -objective(x),
            best_x,
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        best_x = polish_result.x
    except Exception:
        pass

    # Final check and reshape
    best_points = best_x.reshape(n_points, dim)
    
    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        best_points = np.random.uniform(-1, 1, (n_points, dim))

    return best_points

# EVOLVE-BLOCK-END
