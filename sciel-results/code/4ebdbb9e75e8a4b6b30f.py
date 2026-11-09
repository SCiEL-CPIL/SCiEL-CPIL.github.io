# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 580  # Leave 50s margin from 630s to be safe
    
    def objective(points_flat):
        """
        Objective function to maximize: A_min / A_hull.
        points_flat: shape (26,) representing 13 (x,y) points.
        """
        # Check time budget
        if time.time() - start_time > time_limit:
            return 0.0
            
        points = points_flat.reshape((n, 2))
        
        # Check for valid points (finite, etc.)
        if not np.all(np.isfinite(points)):
            return 0.0

        # Calculate convex hull area
        try:
            hull = ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 0:
                return 0.0
        except Exception:
            return 0.0

        # Calculate min triangle area
        # Vectorized calculation of all 286 triangle areas
        # i, j, k indices for combinations of 3 from 13
        idx = np.indices((n, n, n)).reshape(3, -1).T
        # Filter unique combinations i < j < k
        mask = (idx[:, 0] < idx[:, 1]) & (idx[:, 1] < idx[:, 2])
        idx = idx[mask]
        
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        
        # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
        # Cross product magnitude in 2D is |x1*y2 - x2*y1 + x2*y3 - x3*y2 + x3*y1 - x1*y3|
        # Or simpler: 0.5 * | (p2-p1) x (p3-p1) |
        # Cross product in 2D: (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        
        A_min = np.min(areas)
        
        if A_min <= 0:
            return 0.0
            
        return A_min / A_hull

    def neg_objective(points_flat):
        """
        Negated objective for minimization.
        """
        return -objective(points_flat)

    # Define bounds: [0, 1] x [0, 1] square
    # Differential Evolution works on flat vectors
    bounds = [(0.0, 1.0)] * (n * 2)
    
    # We want to maximize, so we minimize the negative
    # Use Differential Evolution for global search
    # maxiter and popsize are tuned to fit within time limit
    # popsize=15 is moderate, maxiter=200 allows sufficient exploration
    # tol=1e-6 for convergence
    
    try:
        result = differential_evolution(
            neg_objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15, 
            tol=1e-5,
            polish=False # Polish can be slow and risky near boundaries
        )
        
        best_points = result.x.reshape((n, 2))
        
        # Verify the result is valid
        if not np.all(np.isfinite(best_points)):
            # Fallback to random if optimization failed badly
            rng = np.random.default_rng(seed=42)
            return rng.random((n, 2))
            
        return best_points
        
    except Exception:
        # Fallback to random if an error occurs
        rng = np.random.default_rng(seed=42)
        return rng.random((n, 2))

# EVOLVE-BLOCK-END
