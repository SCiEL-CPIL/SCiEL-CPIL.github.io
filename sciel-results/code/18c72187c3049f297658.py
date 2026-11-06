import numpy as np
from scipy.optimize import differential_evolution, minimize
import math

def construct():
    n = 8
    
    def objective(params):
        pts = params.reshape(n, 2)
        # Calculate pairwise distances
        diff = pts[:, np.newaxis] - pts[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid log(0)
        dists = np.maximum(dists, 1e-10)
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        
        # Log product of all ordered pairs
        log_prod = np.sum(np.log(dists[mask]))
        
        # Penalty for diameter > 2
        max_dist = np.max(dists)
        penalty = 0.0
        if max_dist > 2.0:
            # Strong penalty to enforce constraint
            penalty = 1000.0 * (max_dist - 2.0) ** 2
        
        return -log_prod + penalty

    # Bounds for differential evolution
    # Points should be within a reasonable range. Diameter is 2, so coords in [-2, 2] is safe.
    bounds = [(-2.0, 2.0)] * (2 * n)
    
    # Run Differential Evolution
    # Using a smaller population and iterations to fit time limit, but enough for global search
    try:
        de_result = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=15, 
            tol=1e-8, 
            seed=42,
            polish=True, # Use L-BFGS-B to polish the result
            init='sobol'
        )
        
        best_pts = de_result.x.reshape(n, 2)
        
    except Exception:
        # Fallback to regular octagon if DE fails
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_pts = np.array([np.cos(angles), np.sin(angles)]).T

    # Additional local optimization with SLSQP to ensure high precision
    # We use the penalty method again but with SLSQP which handles gradients
    def objective_slsqp(params):
        pts = params.reshape(n, 2)
        diff = pts[:, np.newaxis] - pts[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        dists = np.maximum(dists, 1e-10)
        mask = ~np.eye(n, dtype=bool)
        log_prod = np.sum(np.log(dists[mask]))
        max_dist = np.max(dists)
        penalty = 0.0
        if max_dist > 2.0:
            penalty = 10000.0 * (max_dist - 2.0) ** 2
        return -log_prod + penalty

    try:
        res = minimize(
            objective_slsqp, 
            best_pts.flatten(), 
            method='L-BFGS-B', 
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        if res.success:
            best_pts = res.x.reshape(n, 2)
    except Exception:
        pass

    # Scale to diameter 2
    diff = best_pts[:, np.newaxis] - best_pts[np.newaxis, :]
    dists = np.linalg.norm(diff, axis=2)
    max_d = np.max(dists)
    
    if max_d > 1e-6:
        best_pts = best_pts * (2.0 / max_d)
        
    return best_pts.tolist()
