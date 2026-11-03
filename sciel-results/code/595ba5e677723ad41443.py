import numpy as np
import time
import scipy.optimize as so

N = 50

# EVOLVE-BLOCK-START

def _generate_points(params):
    """
    Generate 50 points from 8 parameters:
    r1, r2, r3, r4, theta1, theta2, theta3, theta4
    Structure: 1 center, 6, 12, 15, 16 points in rings.
    """
    r1, r2, r3, r4, t1, t2, t3, t4 = params
    counts = [1, 6, 12, 15, 16]
    radii = [0.0, r1, r2, r3, r4]
    thetas = [0.0, t1, t2, t3, t4]
    
    points = []
    for i, (count, r, t) in enumerate(zip(counts, radii, thetas)):
        if count == 0:
            continue
        if i == 0:
            points.append([0.0, 0.0])
        else:
            angles = t + np.arange(count) * (2.0 * np.pi / count)
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            points.append(np.column_stack((x, y)))
            
    return np.vstack(points)

def _min_distance(points):
    """
    Calculate the minimum distance between any two points or to the boundary.
    """
    # Distance to boundary
    dist_bound = 1.0 - np.linalg.norm(points, axis=1)
    
    # Pairwise distances
    # Use broadcasting for vectorization
    # diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # dists = np.linalg.norm(diff, axis=-1)
    # np.fill_diagonal(dists, np.inf)
    # min_pair = np.min(dists)
    
    # More memory efficient:
    min_pair = np.inf
    for i in range(len(points)):
        # Distance from i to all j > i
        diffs = points[i+1:] - points[i]
        dists = np.linalg.norm(diffs, axis=1)
        if len(dists) > 0:
            min_pair = min(min_pair, np.min(dists))
            
    return min(min_pair, np.min(dist_bound))

def _objective(params):
    """
    Objective function for minimization.
    Returns negative of min distance.
    """
    points = _generate_points(params)
    return -_min_distance(points)

def _smooth_objective(params, beta=10.0):
    """
    Smooth approximation of the min distance for global optimization.
    """
    points = _generate_points(params)
    
    # Distances to boundary
    dist_bound = 1.0 - np.linalg.norm(points, axis=1)
    
    # Pairwise distances
    min_pair_per_point = np.inf * np.ones(len(points))
    for i in range(len(points)):
        diffs = points[i+1:] - points[i]
        dists = np.linalg.norm(diffs, axis=1)
        if len(dists) > 0:
            min_pair_per_point[i] = np.min(dists)
        # Also need dist to previous points? No, min_pair_per_point[i] should be min dist to ANY other point.
        # The loop above only checks i to j>i.
        # We need to handle the full min for each point.
        
    # Recalculating min dist for each point correctly
    min_dist_per_point = np.inf * np.ones(len(points))
    for i in range(len(points)):
        diffs = points - points[i]
        dists = np.linalg.norm(diffs, axis=1)
        dists[i] = np.inf # Exclude self
        min_dist_per_point[i] = np.min(dists)
        
    # Combine with boundary distance
    all_dists = np.minimum(min_dist_per_point, dist_bound)
    
    # Log-sum-exp approximation of min
    # -1/beta * log(sum(exp(-beta * dist)))
    # Use stable logsumexp
    shifted = np.min(all_dists)
    log_sum = np.log(np.sum(np.exp(-beta * (all_dists - shifted))))
    smooth_min = shifted - (1.0 / beta) * log_sum
    
    return -smooth_min

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    # Bounds for 8 parameters: r1, r2, r3, r4, t1, t2, t3, t4
    bounds = [
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
        (0.0, 2.0 * np.pi), (0.0, 2.0 * np.pi), (0.0, 2.0 * np.pi), (0.0, 2.0 * np.pi)
    ]
    
    # Initial guess
    # Rough estimate of radii for 50 circles
    # Area ~ 50 * pi * r^2 <= pi * 1^2 => r <= 1/sqrt(50) ~ 0.14
    # Known r ~ 0.125
    # Ring radii roughly proportional to sqrt(k)
    init_r1 = 0.12
    init_r2 = 0.24
    init_r3 = 0.36
    init_r4 = 0.48
    init_params = np.array([init_r1, init_r2, init_r3, init_r4, 0, 0, 0, 0])
    
    # Differential Evolution on 8D space
    # Using a smooth objective to help DE navigate
    best_result = None
    best_score = -np.inf
    
    # Run DE
    # maxiter is a soft limit; we check time
    de_result = so.differential_evolution(
        _smooth_objective, 
        bounds, 
        args=(50.0,), # beta
        seed=42,
        maxiter=100,
        popsize=20,
        tol=1e-6,
        polish=False, # We will do our own polish
        init='sobol'
    )
    
    if time.time() - start_time > timeout:
        # Fallback if DE is too slow (unlikely for 8D)
        points = _generate_points(init_params)
        return points
    
    best_params_8d = de_result.x
    best_points = _generate_points(best_params_8d)
    best_score = _min_distance(best_points)
    
    # Local Polish on 100D space
    # Start from the best 8D-derived points
    # Use L-BFGS-B with a smooth objective on the full 100D space
    # Or Powell on the exact objective
    
    # Let's try polishing the 100 points directly
    # Objective: -min_distance(points)
    # Bounds for each point: x in [-1, 1], y in [-1, 1]
    # But we must ensure they stay inside the unit circle.
    # L-BFGS-B with bounds is good, but the constraint |p| <= 1 is non-linear.
    # We can use the smooth objective which penalizes going outside.
    
    # Reshape points to 100D vector
    x0 = best_points.flatten()
    
    # Define objective for 100D
    def obj_100d(x):
        p = x.reshape(-1, 2)
        return -_min_distance(p)
        
    def smooth_obj_100d(x, beta=100.0):
        p = x.reshape(-1, 2)
        # Dist to boundary
        dist_bound = 1.0 - np.linalg.norm(p, axis=1)
        
        # Min dist per point
        min_dist_per_point = np.inf * np.ones(len(p))
        # Vectorized pairwise min is tricky without full matrix, but 50x50 is small
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_dist_per_point = np.min(dists, axis=1)
        
        all_dists = np.minimum(min_dist_per_point, dist_bound)
        
        # Smooth min
        shifted = np.min(all_dists)
        log_sum = np.log(np.sum(np.exp(-beta * (all_dists - shifted))))
        smooth_min = shifted - (1.0 / beta) * log_sum
        
        return -smooth_min

    # Polish with L-BFGS-B
    # Use increasing beta
    current_x = x0
    for beta in [10.0, 50.0, 100.0, 500.0]:
        res = so.minimize(
            smooth_obj_100d, 
            current_x, 
            args=(beta,),
            method='L-BFGS-B',
            bounds=[(-1.0, 1.0)] * 100,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        if res.fun < -np.inf: # Always true, but check validity
            current_x = res.x
            
        if time.time() - start_time > timeout - 5.0:
            break
            
    # Final check and return
    final_points = current_x.reshape(-1, 2)
    
    # Ensure validity: clip to unit circle? No, if it's outside, score is 0.
    # But L-BFGS-B with bounds keeps it in [-1,1] square.
    # We need to ensure it's inside the circle.
    # The objective function penalizes distance to boundary, so it should stay inside.
    # However, numerical errors might push it slightly out.
    # Let's check and project if necessary, or just return.
    # The evaluator checks |c_i| < 1.
    
    # One more polish with exact objective using Powell (robust to non-smoothness)
    # Powell is slow but we have time.
    if time.time() - start_time < timeout - 10.0:
        res_powell = so.minimize(
            obj_100d, 
            current_x,
            method='Powell',
            options={'maxiter': 50, 'ftol': 1e-10}
        )
        if res_powell.fun < obj_100d(current_x):
            current_x = res_powell.x
            
    final_points = current_x.reshape(-1, 2)
    
    # Final validation: ensure all points are strictly inside
    # If any point is outside, move it in slightly? 
    # Better: The smooth objective should keep them in.
    
    return final_points

# EVOLVE-BLOCK-END
