# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _raw_metric(points: np.ndarray) -> float:
    """Compute the actual metric (d_min / d_max)^2."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def _smooth_metric(points: np.ndarray, beta: float) -> float:
    """
    Compute the negative smooth surrogate for the metric.
    We minimize the negative of (softmin / softmax)^2.
    Using log-sum-exp for stability.
    """
    dists = pdist(points)
    
    # Softmin: -1/beta * log(sum(exp(-beta * d)))
    # To avoid overflow/underflow, we use the log-sum-exp trick:
    # log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    
    # For softmin(d, beta):
    # Let y = -beta * d
    # softmin = -1/beta * (max(y) + log(sum(exp(y - max(y)))))
    y_min = -beta * dists
    max_y_min = np.max(y_min)
    if not np.isfinite(max_y_min):
        return 0.0
    log_sum_exp_min = max_y_min + np.log(np.sum(np.exp(y_min - max_y_min)))
    soft_min = -1.0 / beta * log_sum_exp_min

    # For softmax(d, beta):
    # Let z = beta * d
    # softmax = 1/beta * (max(z) + log(sum(exp(z - max(z)))))
    z_max = beta * dists
    max_z_max = np.max(z_max)
    if not np.isfinite(max_z_max):
        return 0.0
    log_sum_exp_max = max_z_max + np.log(np.sum(np.exp(z_max - max_z_max)))
    soft_max = 1.0 / beta * log_sum_exp_max

    if soft_max == 0:
        return 0.0
        
    ratio = soft_min / soft_max
    # We want to maximize ratio^2, so minimize -ratio^2
    return -ratio * ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Finds 14 points in 3D maximizing (d_min / d_max)^2.
    """
    n_points = 14
    dim = 3
    n_vars = n_points * dim
    
    start_time = time.time()
    deadline = start_time + 580.0 # 20s safety margin
    
    best_points = None
    best_score = -1.0
    
    # Beta schedule for annealing the soft-min/max
    betas = [5, 10, 20, 50, 100, 200]
    
    # Number of restarts to attempt
    max_restarts = 100 
    
    # Pre-allocate for speed if needed, but np.random is fast enough
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialization: Random points on a sphere of radius 1
        # This helps avoid the trivial collapsed solution
        np.random.seed(42 + r)
        points = np.random.randn(n_points, dim)
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Flatten for optimizer
        x0 = points.flatten()
        
        # Optimization loop with increasing beta
        current_x = x0.copy()
        
        for beta in betas:
            if time.time() > deadline:
                break
                
            # Define the objective function for this beta
            # We also center the points to remove translational invariance
            # and normalize scale to prevent drift
            def objective(x):
                p = x.reshape(n_points, dim)
                # Center
                p = p - np.mean(p, axis=0)
                # Scale to prevent numerical issues (optional, but good for stability)
                # The metric is scale invariant, so this doesn't change the value
                # but helps the optimizer.
                norm = np.linalg.norm(p)
                if norm < 1e-10:
                    return 0.0
                p = p / norm
                return _smooth_metric(p, beta)
            
            # Gradient is not provided, so we use numerical differentiation (default)
            # L-BFGS-B is robust to this for small dimensions
            try:
                res = minimize(
                    objective, 
                    current_x, 
                    method='L-BFGS-B', 
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.success or res.fun < objective(current_x):
                    current_x = res.x
            except Exception:
                # If optimization fails, keep current_x
                pass
        
        # Evaluate the final candidate from this restart
        # Re-center and normalize for final metric calculation
        final_points = current_x.reshape(n_points, dim)
        final_points = final_points - np.mean(final_points, axis=0)
        norm = np.linalg.norm(final_points)
        if norm > 1e-10:
            final_points = final_points / norm
            
        score = _raw_metric(final_points)
        
        if score > best_score:
            best_score = score
            best_points = final_points.copy()
            
        # Early stopping if we hit a very high score
        # Theoretical max for N=14 in 3D is unknown but likely < 1.0
        # If we get close to 1.0, we can stop.
        if best_score > 0.95:
            break

    if best_points is None:
        # Fallback: return random points if something went wrong
        np.random.seed(42)
        best_points = np.random.randn(n_points, dim)
        
    return best_points

# EVOLVE-BLOCK-END
