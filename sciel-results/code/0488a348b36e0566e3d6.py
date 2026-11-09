# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 600.0  # Safety margin for 630s limit
    
    best_score = -1.0
    best_points = None

    # Objective function for minimization
    # We transform the points to have diameter 1, then maximize the minimum distance.
    # This is equivalent to maximizing (d_min / d_max)^2 since d_max becomes 1.
    def objective(x):
        points = x.reshape(n, d)
        # Center the points
        points = points - np.mean(points, axis=0)
        
        # Calculate distances
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
            
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
            
        # Normalize so that d_max = 1
        points_norm = points / d_max
        
        # Recalculate distances on normalized points
        dists_norm = pdist(points_norm)
        d_min = np.min(dists_norm)
        
        # We want to maximize d_min, so minimize -d_min
        # Using squared distance or just distance? 
        # Metric is (d_min/d_max)^2. With d_max=1, it's d_min^2.
        # Minimizing -d_min is often smoother than -d_min^2? 
        # Actually, d_min is piecewise linear (in terms of squared dist) or sqrt of quadratic.
        # Let's minimize -d_min to keep it simple. The square is monotonic.
        return -d_min

    # Random restarts
    np.random.seed(42)
    restarts = 0
    while time.time() - start_time < time_limit:
        # Generate random initial points
        # Random points in a cube or sphere
        points_init = np.random.randn(n, d)
        
        # Optimization
        try:
            result = minimize(objective, points_init.flatten(), method='Nelder-Mead', 
                              options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
            
            if result.success or result.fun < best_score:
                # Calculate actual score for the best result to be sure
                pts = result.x.reshape(n, d)
                dists = pdist(pts)
                d_min = np.min(dists)
                d_max = np.max(dists)
                if d_max > 1e-10:
                    score = (d_min / d_max) ** 2
                else:
                    score = 0.0
                
                if score > best_score:
                    best_score = score
                    best_points = pts
        except Exception:
            pass
        
        restarts += 1
        # If we've done enough restarts, we might break early if time is tight
        # But Nelder-Mead is fast. Let's keep going until time limit.

    if best_points is None:
        # Fallback to random if optimization failed completely
        best_points = np.random.randn(n, d)

    return best_points

# EVOLVE-BLOCK-END
