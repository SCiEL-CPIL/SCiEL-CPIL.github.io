import numpy as np
from scipy.optimize import dual_annealing
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Initial guess: Regular octagon with diameter 2
    # Radius R such that 2R = 2 => R = 1
    # Points are at angles 2*pi*k/8
    k = np.arange(8)
    angles = 2 * np.pi * k / 8
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # We want to maximize sum(log(d_ij)) subject to d_ij <= 2
    # Let's minimize -sum(log(d_ij)) + penalty
    
    def objective(points_flat):
        if time.time() > deadline:
            # Return a very large value to stop optimization? 
            # dual_annealing might not respect this perfectly, but we can check later.
            pass
            
        points = points_flat.reshape(8, 2)
        # Compute pairwise distances
        # d[i, j] = ||points[i] - points[j]||
        # Use broadcasting
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Mask diagonal
        mask = ~np.eye(8, dtype=bool)
        d = dists[mask]
        
        # If any distance is 0, return large penalty
        if np.any(d < 1e-10):
            return 1e10
            
        # Log product term
        log_sum = np.sum(np.log(d))
        
        # Constraint penalty: d <= 2
        # Use a quadratic penalty for violations
        violations = np.maximum(0, d - 2.0)
        penalty = 1000.0 * np.sum(violations ** 2)
        
        # We want to maximize log_sum, so minimize -log_sum
        return -log_sum + penalty

    # Bounds for points
    # Since diameter is 2, points must be within a circle of radius 2? 
    # Actually, if diameter is 2, the whole set fits in a disk of radius 2? 
    # No, the diameter is the max distance between any two points.
    # The set is contained in a disk of radius 2 centered at the midpoint of the diameter pair?
    # Yes, by Jung's theorem, any set of diameter D in R^2 is contained in a disk of radius D/sqrt(3).
    # But simpler: if max distance is 2, no point can be further than 2 from any other.
    # So all points are in the intersection of disks of radius 2 centered at each point.
    # A safe bound is [-2, 2] for x and y.
    bounds = [(-2.0, 2.0)] * 16
    
    best_result = None
    best_val = float('inf')
    
    # Run dual_annealing
    # maxiter and maxfun can be set high, but we will rely on time
    try:
        res = dual_annealing(
            objective, 
            x0=x0.flatten(), 
            bounds=bounds, 
            maxiter=100, 
            maxfun=10000,
            seed=42
        )
        
        if time.time() < deadline:
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x
    except Exception:
        pass
    
    # If we have time, try a few more restarts with different seeds or initial points
    # But dual_annealing is already stochastic.
    
    # Fallback to regular octagon if optimization failed or didn't improve
    if best_result is None:
        best_result = x0.flatten()
        
    points = best_result.reshape(8, 2)
    
    # Verify and normalize to diameter 2 just in case
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    d_max = np.max(dists)
    
    if d_max > 1e-10:
        points = points * (2.0 / d_max)
        
    return points.tolist()
