"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    # Keep a safety margin of 10s before the 120s limit
    deadline = start_time + 110.0
    
    # 1. Generate Initial Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    best_centers = grid[:N].astype(np.float64)
    best_score = -1.0
    
    def get_score(centers):
        # Calculate minimum pair distance
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Calculate minimum wall distance
        min_wall_dist = np.min(0.5 - np.abs(centers))
        
        return min(min_pair_dist / 2.0, min_wall_dist)
    
    # Initial score
    best_score = get_score(best_centers)
    
    # 2. Local Optimization Loop with Restarts
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize current state
        if restart == 0:
            centers = best_centers.copy()
        else:
            # Perturb the best solution found so far
            centers = best_centers.copy()
            noise_scale = 0.1 / (restart + 1)
            centers += np.random.normal(0, noise_scale, centers.shape)
            centers = np.clip(centers, -0.5, 0.5)
        
        # Run Repulsion Optimization
        # We use a decaying learning rate
        steps = 2000
        learning_rate = 0.05
        decay = 0.999
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Compute pairwise differences and distances
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # Avoid division by zero
            safe_dists = np.maximum(dists, 1e-6)
            
            # Repulsive forces: F = (c_i - c_j) / d^2
            # This pushes spheres apart.
            inv_dists_sq = 1.0 / (safe_dists ** 2)
            # Mask self-interaction
            inv_dists_sq[~np.isfinite(inv_dists_sq)] = 0.0
            
            forces = np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
            
            # Wall Repulsion
            # We want to keep spheres away from walls.
            # If a sphere is too close to a wall, push it towards the center.
            # Threshold: if dist_to_wall < r_current * 0.5, push.
            # Estimate current radius
            min_pair_dist = np.min(dists)
            min_wall_dist = np.min(0.5 - np.abs(centers))
            current_r = min(min_pair_dist / 2.0, min_wall_dist)
            
            margin = max(current_r * 0.5, 0.01)
            
            wall_forces = np.zeros_like(centers)
            
            # Check upper boundaries
            # If c > 0.5 - margin, push left
            mask_u = centers > (0.5 - margin)
            wall_forces[mask_u] -= (centers[mask_u] - (0.5 - margin)) * 5.0
            
            # Check lower boundaries
            # If c < -0.5 + margin, push right
            mask_l = centers < (-0.5 + margin)
            wall_forces[mask_l] -= (centers[mask_l] - (-0.5 + margin)) * 5.0
            
            forces += wall_forces
            
            # Update positions
            centers += learning_rate * forces
            
            # Clip to box
            centers = np.clip(centers, -0.5, 0.5)
            
            # Decay learning rate
            learning_rate *= decay
        
        # Check if we found a better solution
        current_score = get_score(centers)
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
        
        # If time is running out, break
        if time.time() > deadline - 5.0:
            break
            
    return best_centers
# EVOLVE-BLOCK-END
