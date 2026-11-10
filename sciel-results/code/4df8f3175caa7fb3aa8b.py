import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

def _score(points):
    """
    Calculate the metric: (d_min / d_max)^2.
    Returns 0.0 if invalid.
    """
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
        
    d_min = np.min(dists)
    d_max = np.max(dists)
    
    if d_max < 1e-12:
        return 0.0
        
    ratio = d_min / d_max
    return ratio * ratio

def _fibonacci_sphere(n, R=1.0):
    """
    Generate n points on a sphere using a Fibonacci lattice.
    """
    golden_ratio = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    # Avoid exact poles to prevent symmetry issues that might trap optimizers
    # z ranges from -1 to 1
    z = 1 - (2 * i + 1) / n
    r = np.sqrt(np.clip(1 - z**2, 0, 1))
    theta = 2 * np.pi * i / golden_ratio
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    points = np.column_stack((x, y, z)) * R
    return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Finds 14 points in 3D maximizing (d_min / d_max)^2.
    """
    start_time = time.time()
    deadline = start_time + 550.0  # Safety margin for 600s limit
    
    n = 14
    d = 3
    
    # Initial best solution: Fibonacci sphere
    best_points = _fibonacci_sphere(n, R=1.0)
    best_score = _score(best_points)
    
    # Optimization strategy: Multi-start local optimization
    # We will perturb the initial points and run a local minimizer on -score (minimizing negative score)
    
    np.random.seed(42) # Fixed seed for reproducibility in restarts if needed, but we use random perturbations
    
    max_restarts = 50 # Cap on restarts, but also limited by time
    
    for i in range(max_restarts):
        if time.time() > deadline - 5.0: # Stop if less than 5s left
            break
            
        # Create a perturbed version of the initial sphere
        # Perturbation magnitude can be small to stay in the basin, or larger for exploration
        # We try a mix: small perturbations for refinement, larger for exploration
        if i < 10:
            noise_scale = 0.1
        elif i < 30:
            noise_scale = 0.5
        else:
            noise_scale = 1.0
            
        # Start from the best known point so far, or the initial sphere?
        # Starting from the best known point allows for further refinement.
        # Starting from the initial sphere allows for exploration of different basins.
        # Let's alternate or just use the initial sphere + noise for exploration,
        # and occasionally refine the best point.
        
        if i % 5 == 0 and i > 0:
            # Refine the current best
            start_pts = best_points.copy()
            noise = np.random.randn(n, d) * 0.01
        else:
            # Explore from initial structure
            start_pts = _fibonacci_sphere(n, R=1.0)
            noise = np.random.randn(n, d) * noise_scale
            
        current_pts = start_pts + noise
        
        # Optimization
        # Flatten points for optimization
        def objective(x_flat):
            pts = x_flat.reshape(n, d)
            return -_score(pts)
            
        x0 = current_pts.flatten()
        
        try:
            # Powell is good for non-smooth functions and doesn't need gradients
            # Nelder-Mead is also an option. Powell tends to be more robust in higher dimensions 
            # but Nelder-Mead is faster per step. Let's use Powell with maxiter limit.
            res = minimize(objective, x0, method='Powell', options={'maxiter': 1000, 'xtol': 1e-8, 'ftol': 1e-8})
            
            if res.success or res.fun < -best_score:
                candidate_pts = res.x.reshape(n, d)
                candidate_score = -res.fun
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_pts
                    
        except Exception:
            pass
            
    # Final polish phase: use SLSQP with tight tolerance on the best solution found
    # This addresses the directive to investigate a final high-precision polish pass
    if time.time() < deadline - 10.0:
        def objective_polish(x_flat):
            pts = x_flat.reshape(n, d)
            return -_score(pts)
            
        x0 = best_points.flatten()
        
        try:
            # SLSQP can handle the non-smoothness better with careful settings
            # We minimize -score, so we want to maximize score
            res = minimize(objective_polish, x0, method='SLSQP', 
                          options={'maxiter': 1000, 'ftol': 1e-12, 'eps': 1e-8})
            
            if res.fun < -best_score:
                candidate_pts = res.x.reshape(n, d)
                candidate_score = -res.fun
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_pts
                    
        except Exception:
            pass
        
        # Try another polish with L-BFGS-B if time permits
        if time.time() < deadline - 5.0:
            try:
                res = minimize(objective_polish, best_points.flatten(), method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
                
                if res.fun < -best_score:
                    candidate_pts = res.x.reshape(n, d)
                    candidate_score = -res.fun
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_points = candidate_pts
                        
            except Exception:
                pass
                
    return best_points


