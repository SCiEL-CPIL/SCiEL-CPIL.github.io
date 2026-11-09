import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin for 150s limit
    
    # 1. Initial Configuration: Simple Cubic Grid
    # Use a 5x5x5 grid (125 points) and select 100.
    # To maximize potential, select points that are not corners/edges if possible?
    # Actually, for local optimization, starting from a dense grid is fine.
    # Let's generate a 5x5x5 grid.
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # We have 125 points. We need 100.
    # Randomly removing points might leave clumps. 
    # Better: Take the 100 points with the largest "interior-ness" or just random?
    # Let's try to keep the structure somewhat regular.
    # We can just take the first 100, but that biases towards one corner.
    # Let's shuffle and take 100, or better, remove the 25 points that are furthest from the center? No, that removes interior.
    # Remove points that are on the boundary? 
    # Corners (8), Edges (36), Faces (54), Interior (27).
    # If we remove 25, we can keep all 27 interior? No, 27 < 25.
    # So we must remove some interior or face points.
    # Actually, just random selection from the grid is a valid start.
    indices = np.random.choice(125, 100, replace=False)
    centers = grid[indices].copy()
    
    # 2. Local Optimization
    # We want to maximize min(dist(i,j), dist(i,wall))
    
    def get_score(centers):
        # Compute min pair distance
        dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Compute min distance to wall
        # Centers are in [-0.5, 0.5]
        # dist to wall = 0.5 - |c|
        min_wall_dist = np.min(0.5 - np.abs(centers))
        
        return min(min_pair_dist, min_wall_dist)

    current_score = get_score(centers)
    
    # Simple Hill Climbing / Gradient Ascent
    # We will iteratively move points away from their closest constraints.
    
    for iteration in range(10000):
        if time.time() > start_time + timeout:
            break
            
        # Calculate forces/gradients for each point
        # For each point i, find the closest point j and closest wall.
        # Move i away from j and wall.
        
        # Efficient calculation of closest neighbors is O(N^2) per step, which is 100^2 = 10k ops.
        # 10k * 10k iterations = 100M ops. Might be too slow in Python.
        # Let's optimize the inner loop or reduce iterations.
        
        # Alternative: Move all points simultaneously based on current gradients.
        
        # 1. Find closest neighbor for each point
        dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Closest neighbor index for each i
        closest_idx = np.argmin(dists, axis=1)
        closest_dist = np.min(dists, axis=1)
        
        # 2. Find closest wall for each point
        # Walls are at +/- 0.5
        # Distance to wall for each coordinate
        # d_x = min(c_x + 0.5, 0.5 - c_x)
        # d_y = min(c_y + 0.5, 0.5 - c_y)
        # d_z = min(c_z + 0.5, 0.5 - c_z)
        # min_wall_dist_i = min(d_x, d_y, d_z)
        
        dx = np.minimum(centers[:, 0] + 0.5, 0.5 - centers[:, 0])
        dy = np.minimum(centers[:, 1] + 0.5, 0.5 - centers[:, 1])
        dz = np.minimum(centers[:, 2] + 0.5, 0.5 - centers[:, 2])
        
        min_wall_dist = np.minimum(np.minimum(dx, dy), dz)
        
        # Determine the dominant constraint for each point
        # If closest_dist < min_wall_dist, move away from neighbor
        # Else move away from wall
        
        step_size = 0.01 # Initial step size, can be adaptive
        
        for i in range(N):
            if closest_dist[i] < min_wall_dist[i]:
                # Move away from closest neighbor
                j = closest_idx[i]
                diff = centers[i] - centers[j]
                norm = np.linalg.norm(diff)
                if norm > 1e-6:
                    move = (diff / norm) * step_size
                    centers[i] += move
            else:
                # Move away from wall
                # Determine which wall is closest
                # Check x, y, z distances
                d = np.array([dx[i], dy[i], dz[i]])
                min_d = np.min(d)
                
                # Direction away from wall
                # If c_x is close to 0.5, move in -x. If close to -0.5, move in +x.
                move = np.zeros(3)
                if d[0] == min_d:
                    if centers[i, 0] > 0: move[0] = -1
                    else: move[0] = 1
                if d[1] == min_d:
                    if centers[i, 1] > 0: move[1] = -1
                    else: move[1] = 1
                if d[2] == min_d:
                    if centers[i, 2] > 0: move[2] = -1
                    else: move[2] = 1
                    
                # Normalize if multiple walls are equally close (corners/edges)
                norm = np.linalg.norm(move)
                if norm > 0:
                    move = (move / norm) * step_size
                    centers[i] += move
            
            # Clip to cube
            centers[i] = np.clip(centers[i], -0.5, 0.5)
            
        # Optional: Random perturbation to escape local minima
        if iteration % 100 == 0:
             # Small random jitter
             centers += np.random.normal(0, 0.005, (N, 3))
             centers = np.clip(centers, -0.5, 0.5)

    return centers
