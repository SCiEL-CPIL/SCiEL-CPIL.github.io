# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def _soft_min(dists, K):
    """
    Compute the soft minimum of an array of distances.
    Approximates min(dists) as K -> infinity.
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


def _objective(points_flat, K=50.0, lambda_sphere=100.0):
    """
    Objective function to minimize.
    We want to maximize the minimum distance, so we minimize -soft_min(distances).
    We also want points to be on the unit sphere, so we add a penalty term.
    """
    n = 14
    points = points_flat.reshape(n, 3)
    
    # Calculate pairwise distances
    dists = pdist(points)
    
    # Soft minimum of distances (we want to maximize this, so minimize negative)
    soft_min_dist = _soft_min(dists, K)
    
    # Sphere constraint penalty: sum(||p_i||^2 - 1)^2
    norms_sq = np.sum(points**2, axis=1)
    sphere_penalty = np.sum((norms_sq - 1.0)**2)
    
    # Total objective: minimize -soft_min + penalty
    return -soft_min_dist + lambda_sphere * sphere_penalty


def _evaluate(points: np.ndarray) -> float:
    """Compute the metric (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    n = 14
    d = 3
    
    best_points = None
    best_score = 0.0
    
    np.random.seed(42)
    
    # Number of restarts for the gradient-based optimizer
    # L-BFGS-B is fast, so we can afford many restarts
    n_restarts = 200
    
    for i in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random initialization on the sphere
        points = np.random.randn(n, d)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Flatten for scipy.optimize
        x0 = points.flatten()
        
        try:
            res = minimize(
                _objective, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.status == 0:
                optimized_points = res.x.reshape(n, d)
                
                # Normalize to unit sphere to clean up numerical drift
                norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
                # Avoid division by zero
                norms[norms < 1e-10] = 1.0
                optimized_points = optimized_points / norms
                
                score = _evaluate(optimized_points)
                
                if score > best_score:
                    best_score = score
                    best_points = optimized_points.copy()
                    
        except Exception:
            continue

    if best_points is None:
        # Fallback: random points
        points = np.random.randn(n, d)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        return points
    
    return best_points


# EVOLVE-BLOCK-END
