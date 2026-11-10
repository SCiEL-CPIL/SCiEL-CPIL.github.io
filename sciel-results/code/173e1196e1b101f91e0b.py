# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    
    # Objective function: Minimize negative of (d_min / d_max)^2
    def objective(flat_points):
        pts = flat_points.reshape(n, 2)
        
        # Calculate pairwise distances
        dists = pdist(pts)
        
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0:
            return 0.0
            
        ratio = (d_min / d_max) ** 2
        return -ratio

    # Define bounds for each coordinate
    # We use a box of [-1.5, 1.5] to allow enough space for 16 points
    # while keeping the scale manageable for the optimizer
    bounds = [(-1.5, 1.5)] * (2 * n)
    
    # Use Differential Evolution for global optimization
    # popsize=15, maxiter=200, tol=1e-8
    # This is a robust global method for non-smooth functions
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=15, 
            tol=1e-8,
            seed=42,
            polish=True  # Use L-BFGS-B for final polishing
        )
        
        best_points = res.x.reshape(n, 2)
        
        # Center the points to remove translational invariance
        best_points -= best_points.mean(axis=0)
        
        # Check for validity
        if not np.all(np.isfinite(best_points)):
             raise ValueError("Non-finite points found")
             
        return best_points
    except Exception as e:
        # Fallback to a known good configuration if optimization fails
        # 4x4 grid centered
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        X, Y = np.meshgrid(x, y)
        points = np.column_stack((X.ravel(), Y.ravel()))
        return points

# EVOLVE-BLOCK-END
