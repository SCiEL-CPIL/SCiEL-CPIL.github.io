import numpy as np
import time
import random

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_radius = 0.0
    best_points = None
    
    # Initialize with a grid
    # 4x4x4 = 64 points. We need 50.
    # Or 3x4x5 = 60 points.
    # Let's use a 4x4x4 grid, take first 50.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # We will run multiple restarts with different jittering
    for restart in range(20): # Max restarts, break if time is up
        if time.time() > deadline:
            break
            
        # Create jittered initial points
        # Use a subset of the grid or random points?
        # Grid is better start.
        # Shuffle and take N
        idx = np.random.choice(len(grid), N, replace=False)
        points = grid[idx].copy()
        
        # Add small jitter to break symmetry
        points += np.random.uniform(-0.01, 0.01, points.shape)
        
        # Relaxation
        alpha = 0.05
        for step in range(2000):
            if time.time() > deadline:
                break
                
            # Calculate distances
            # Pairwise distances
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            
            # Wall distances
            # dist to -0.5 and 0.5
            wall_dist = np.minimum(0.5 - points, points + 0.5) # shape (N, 3)
            min_wall_dist = np.min(wall_dist, axis=1) # shape (N,)
            
            # Current radius
            min_pair_dist = np.min(dist)
            r = min(min_pair_dist / 2.0, np.min(min_wall_dist))
            
            if r > best_radius:
                best_radius = r
                best_points = points.copy()
            
            # Calculate forces
            # Repulsion from other spheres
            # F_ij = (c_i - c_j) / dist_ij^2
            # We want to move i away from j.
            # Vector is points[i] - points[j].
            # Force on i from j is proportional to (points[i] - points[j]) / dist^2
            # But we only care about close neighbors? No, all.
            # To avoid division by zero or large forces, we can use a soft potential.
            
            # Efficient vectorized force calculation
            # diff is (N, N, 3)
            # dist is (N, N)
            # We want force_i = sum_j (diff[i,j] / dist[i,j]^2)
            # Note: diff[i,j] = c_i - c_j.
            
            # Avoid division by zero (diagonal)
            # dist has inf on diagonal.
            inv_dist_sq = 1.0 / (dist * dist)
            inv_dist_sq[dist == np.inf] = 0.0
            
            # Force from spheres
            force_spheres = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
            
            # Force from walls
            # If point is close to wall, push it inwards.
            # x > 0: push left (-1). x < 0: push right (+1).
            # Strength: 1 / wall_dist^2
            force_walls = np.zeros_like(points)
            for k in range(3):
                # Distance to right wall (0.5)
                d_r = 0.5 - points[:, k]
                # Distance to left wall (-0.5)
                d_l = points[:, k] + 0.5
                
                # If d_r is small, push left (-1)
                # If d_l is small, push right (+1)
                # Use a threshold to avoid huge forces
                # Actually, just use 1/d^2 for the closest wall?
                # Or sum of forces from both walls?
                # Let's use the closest wall force.
                
                min_d = np.minimum(d_r, d_l)
                # Force direction
                # If d_r < d_l, direction is -1 (away from right wall)
                # If d_l < d_r, direction is +1 (away from left wall)
                dir = np.where(d_r < d_l, -1.0, 1.0)
                
                # Magnitude
                mag = 1.0 / (min_d * min_d)
                
                # Cap magnitude to prevent instability
                mag = np.minimum(mag, 10.0)
                
                force_walls[:, k] = dir * mag
            
            total_force = force_spheres + force_walls
            
            # Update positions
            points += alpha * total_force
            
            # Clip to bounds
            points = np.clip(points, -0.5, 0.5)
            
            # Decay alpha
            alpha *= 0.999
        
        # End relaxation
    
    if best_points is None:
        # Fallback to grid
        return grid[:N]
        
    return best_points

# EVOLVE-BLOCK-END
