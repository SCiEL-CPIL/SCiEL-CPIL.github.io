import numpy as np

def construct():
    N = 50
    # Initialize with points on a sphere using Fibonacci lattice
    # Radius R chosen to be well inside the cube
    R = 0.45
    
    # Fibonacci lattice
    z = 1 - 2 * np.arange(N) / N
    r = np.sqrt(1 - z**2)
    phi = np.arange(N) * np.pi * (3 - np.sqrt(5))
    
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    centers = np.column_stack((x, y, z)) * R
    
    # Optimization
    # Step size schedule
    max_iter = 5000
    step_size = 0.01
    delta_active = 0.01 # Threshold for "active" constraints
    
    for i in range(max_iter):
        # 1. Compute distances
        # Pairwise distances
        # dist[i, j] = ||centers[i] - centers[j]||
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Wall distances
        # min_k (0.5 - |c_ik|)
        # dist to wall is min(0.5-x, 0.5+x, ...) which is 0.5 - max(|x|, |y|, |z|)
        abs_centers = np.abs(centers)
        max_coord = np.max(abs_centers, axis=1)
        wall_dists = 0.5 - max_coord
        
        # Current radius
        min_pair_dist = np.min(dist[~np.eye(N, dtype=bool)])
        min_wall_dist = np.min(wall_dists)
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        # Identify active constraints
        # Pairs with dist < 2 * (current_r + delta)
        # Walls with dist < current_r + delta
        
        active_pairs = np.argwhere(dist < 2 * (current_r + delta_active))
        active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]] # i < j
        
        active_walls = np.where(wall_dists < current_r + delta_active)[0]
        
        # Compute gradient
        grad = np.zeros_like(centers)
        
        # Pair contributions
        for i, j in active_pairs:
            v = centers[i] - centers[j]
            norm_v = np.linalg.norm(v)
            if norm_v > 1e-9:
                v_norm = v / norm_v
                # Increase distance: move i away from j, j away from i
                grad[i] += v_norm
                grad[j] -= v_norm
        
        # Wall contributions
        for i in active_walls:
            # Determine which wall is closest
            # We want to move away from the closest wall
            # If x is close to 0.5, move in -x direction? No, move towards 0.
            # If x is close to -0.5, move in +x direction.
            # Essentially, move towards the center (0,0,0)
            # grad[i] += -centers[i] / ||centers[i]|| ?
            # Or simply: for each coordinate k, if c_ik is the limiting one, move c_ik towards 0.
            # Let's find the coordinate that gave the min wall dist
            # wall_dist_i = 0.5 - max(|c_ix|, |c_iy|, |c_iz|)
            # The limiting coordinate is the one with max abs value.
            limiting_coords = np.argmax(abs_centers[i])
            # Move towards 0 in that coordinate
            # If c_ik > 0, move -1. If c_ik < 0, move +1.
            direction = -np.sign(centers[i, limiting_coords])
            # Add to gradient
            grad[i, limiting_coords] += direction
            # Actually, we can just add -centers[i] to push towards origin, which handles all walls simultaneously if we normalize.
            # But pushing towards origin might reduce distance to other walls? No, origin is the center.
            # Let's use a stronger push towards the center for active wall points.
            # grad[i] += -centers[i] / (np.linalg.norm(centers[i]) + 1e-9)
            
        # Normalize gradient to avoid huge steps
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad /= grad_norm
            
        # Update
        centers += step_size * grad
        
        # Clamp to cube
        centers = np.clip(centers, -0.5, 0.5)
        
        # Decay step size
        step_size *= 0.999
        delta_active *= 0.999
        
    return centers
