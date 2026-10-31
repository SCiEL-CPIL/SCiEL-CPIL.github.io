import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    # Initialize with a jittered grid to avoid perfect symmetry traps
    side = int(np.ceil(np.sqrt(N)))
    points = np.zeros((N, 2))
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            points[idx, 0] = (i + 0.5) / side
            points[idx, 1] = (j + 0.5) / side
            # Add small random jitter
            points[idx, 0] += np.random.uniform(-0.01, 0.01)
            points[idx, 1] += np.random.uniform(-0.01, 0.01)
            idx += 1
            
    # Clamp to [0.01, 0.99] initially
    points = np.clip(points, 0.01, 0.99)
    
    best_points = points.copy()
    best_score = -np.inf
    
    # Precompute pairs for distance calculation
    # We will calculate full distance matrix each step, N=26 is small (26x26)
    
    for step in range(500000): # Max steps
        if time.time() > deadline:
            break
            
        # 1. Calculate current min distance and wall distances
        # Pairwise distances
        # diff[i, j] = points[i] - points[j]
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        min_dist = np.min(dists)
        min_idx = np.argmin(dists)
        i, j = np.unravel_index(min_idx, dists.shape)
        
        # Wall distances
        # min(x, y, 1-x, 1-y)
        wall_dists = np.min(np.stack([
            points[:, 0], 
            points[:, 1], 
            1.0 - points[:, 0], 
            1.0 - points[:, 1]
        ], axis=1), axis=1)
        
        min_wall = np.min(wall_dists)
        min_wall_idx = np.argmin(wall_dists)
        
        current_R = min(min_dist / 2.0, min_wall)
        
        if current_R > best_score:
            best_score = current_R
            best_points = points.copy()
            
        # 2. Determine forces
        # We want to increase the limiting factor.
        # If min_dist/2 < min_wall, the pair is limiting. Push i and j apart.
        # If min_wall < min_dist/2, the wall is limiting. Push the closest circle away from its wall.
        # If they are equal, push both.
        
        move_vec = np.zeros_like(points)
        
        limit_dist = min_dist / 2.0
        limit_wall = min_wall
        
        # Threshold for "active" constraints
        tol = 1e-5
        
        if limit_dist < limit_wall + tol:
            # Pair is limiting (or close to it)
            # Vector from j to i
            v = points[i] - points[j]
            norm_v = np.linalg.norm(v)
            if norm_v > 1e-9:
                v = v / norm_v
            else:
                v = np.array([1.0, 0.0]) # Random if overlapping exactly
            
            # Push i in direction v, j in direction -v
            # Magnitude proportional to step size
            step_size = 0.001
            move_vec[i] += v * step_size
            move_vec[j] -= v * step_size
            
        if limit_wall < limit_dist + tol:
            # Wall is limiting
            k = min_wall_idx
            x, y = points[k]
            
            # Determine which wall is closest
            d_left = x
            d_right = 1.0 - x
            d_down = y
            d_up = 1.0 - y
            
            min_d = min(d_left, d_right, d_down, d_up)
            
            step_size = 0.001
            if min_d == d_left:
                move_vec[k, 0] += step_size
            elif min_d == d_right:
                move_vec[k, 0] -= step_size
            elif min_d == d_down:
                move_vec[k, 1] += step_size
            elif min_d == d_up:
                move_vec[k, 1] -= step_size
                
        # Add small random noise to escape local minima
        # Temperature decreases over time
        temp = 0.01 * (1.0 - step / 500000.0) + 0.001
        noise = np.random.normal(0, temp, size=points.shape)
        
        new_points = points + move_vec + noise
        
        # Keep points inside [0,1] strictly? 
        # The objective function handles the "penalty" via min_wall.
        # But we should avoid points going far outside.
        new_points = np.clip(new_points, -0.1, 1.1)
        
        points = new_points

    # Final calculation of best score
    # Re-evaluate best_points to ensure accuracy
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    wall_dists = np.min(np.stack([
        best_points[:, 0], 
        best_points[:, 1], 
        1.0 - best_points[:, 0], 
        1.0 - best_points[:, 1]
    ], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    
    r = min(min_dist / 2.0, min_wall)
    
    # Construct output
    # The task asks for (x, y, radius).
    # We assume equal radii r.
    result = np.zeros((N, 3))
    result[:, 0] = best_points[:, 0]
    result[:, 1] = best_points[:, 1]
    result[:, 2] = r
    
    return result
