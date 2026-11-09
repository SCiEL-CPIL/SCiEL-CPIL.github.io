# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _soft_min(dists, K):
    """
    Compute the soft minimum of an array of distances.
    """
    # Shift to prevent overflow in exp
    min_val = np.min(dists)
    # We want min(dists), so we minimize the log-sum-exp of -dists
    # log(sum(exp(-K * d))) / K approximates min(d)
    # To avoid underflow/overflow, we subtract the min of the exponents
    # exponents = -K * dists
    # min_exponent = -K * max(dists) ? No.
    # sum(exp(x)) is stable if we subtract max(x).
    # Here x = -K * d. max(x) = -K * min(d).
    # So we subtract -K * min(d) from each term.
    x = -K * dists
    x_max = np.max(x)
    return (x_max + np.log(np.sum(np.exp(x - x_max)))) / K

def _objective(points_flat, K):
    """
    Objective function to minimize.
    Minimizes the soft minimum of normalized pairwise distances.
    """
    points = points_flat.reshape(14, 3)
    
    # Check for NaN or inf
    if not np.all(np.isfinite(points)):
        return 1e10
        
    # Center the points
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Compute pairwise distances
    dists = pdist(points)
    
    if len(dists) == 0:
        return 1e10
        
    d_max = np.max(dists)
    
    if d_max < 1e-10:
        return 1e10
        
    dists_norm = dists / d_max
    
    # We want to maximize min(dists_norm), so we minimize soft_min(dists_norm)
    return _soft_min(dists_norm, K)

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    deadline = time.time() + 600  # 600s limit, leave margin
    
    best_points = None
    best_score = -1.0
    
    # Use multiple random seeds for diversity
    seeds = [42, 123, 456, 789, 101112, 202021, 314159, 271828]
    
    for seed in seeds:
        if time.time() > deadline - 30:
            break
            
        np.random.seed(seed)
        
        # Initialize points on a sphere or randomly in a box
        # Random in box [-1, 1] is fine
        points_init = np.random.uniform(-1, 1, (n, d))
        
        # Optimization
        # We use Powell's method which is derivative-free and robust for this
        # K controls the sharpness of the min approximation
        K = 50.0
        
        try:
            res = minimize(
                _objective, 
                points_init.flatten(), 
                args=(K,),
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-8}
            )
        except Exception:
            continue
            
        opt_points = res.x.reshape(n, d)
        
        # Evaluate actual metric
        if not np.all(np.isfinite(opt_points)):
            continue
            
        # Center
        centroid = np.mean(opt_points, axis=0)
        opt_points = opt_points - centroid
        
        dists = pdist(opt_points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
            continue
            
        score = (d_min / d_max) ** 2
        
        if score > best_score:
            best_score = score
            best_points = opt_points.copy()
            
    # If no valid points found (should not happen), return random
    if best_points is None:
        np.random.seed(0)
        best_points = np.random.randn(n, d)
        
    return best_points

# EVOLVE-BLOCK-END
