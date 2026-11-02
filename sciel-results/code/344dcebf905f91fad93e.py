import numpy as np
import time

def construct():
    N = 100
    # 1. Initialization
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add noise to break symmetry
    np.random.seed(42)
    centers += np.random.normal(0, 0.05, centers.shape)
    centers = np.clip(centers, -0.45, 0.45)
    
    start_time = time.time()
    limit = 115.0
    
    best_centers = centers.copy()
    best_r = -1.0
    
    # Parameters
    step_size = 0.01
    temperature = 0.05
    R_offset = 0.05 # How far beyond min dist to interact
    
    while time.time() - start_time < limit:
        # Calculate distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        d_min = np.min(dists)
        
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        current_r = min(d_min / 2.0, min_wall)
        
        if current_r > best_r:
            best_r = current_r
            best_centers = centers.copy()
            
        # If we've significantly improved, we can continue polishing
        # Target ~ 0.11138
        
        # Determine Interaction Range R
        # Use the current bottleneck + offset
        R = max(d_min, min_wall) + R_offset
        
        # Calculate Forces
        forces = np.zeros_like(centers)
        
        # 1. Sphere-Sphere
        mask = dists < R
        safe_dists = np.where(dists > 0, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        
        # Spring force: k * (R - d)
        # We want to push apart, so force on i is in direction (c_i - c_j)
        # Vector (c_i - c_j) is diff[i,j]
        # So force is positive scalar * diff[i,j]
        
        strengths = np.where(mask, R - dists, 0.0)
        # Sum over j
        forces_ss = np.sum(strengths[:, :, np.newaxis] * diff, axis=1)
        # Note: diff is c_i - c_j. So this pushes i away from j. Correct.
        
        # 2. Walls
        wall_mask = wall_dists < R
        wall_strengths = np.where(wall_mask, R - wall_dists, 0.0)
        # Direction: -sign(c)
        wall_dir = -np.sign(centers)
        forces_walls = np.sum(wall_strengths * wall_dir, axis=1)
        
        total_forces = forces_ss + forces_walls
        
        # Add Noise
        noise = np.random.normal(0, temperature, centers.shape)
        
        # Update
        centers += total_forces * step_size + noise
        
        # Clip
        centers = np.clip(centers, -0.5, 0.5)
        
        # Annealing
        temperature *= 0.99
        step_size *= 0.995
        
        if temperature < 1e-4:
            break
            
    return best_centers
