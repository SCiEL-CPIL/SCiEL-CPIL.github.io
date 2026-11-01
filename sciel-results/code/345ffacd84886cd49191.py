import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin
    
    # 1. Initialize with a grid + noise
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    # Add small noise to break symmetry
    centers += np.random.normal(0, 0.01, (N, 3))
    centers = np.clip(centers, -0.49, 0.49)
    
    def calc_radius(c):
        # Vectorized min distance between pairs
        # This is O(N^2) but vectorized
        # We only need min, so we can do it in chunks or just full matrix for N=100
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_center_dist = np.min(dists)
        
        # Distance to walls
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_center_dist, min_wall_dist) / 2.0

    best_radius = calc_radius(centers)
    best_centers = centers.copy()
    
    step_size = 0.05
    
    while time.time() - start_time < time_limit:
        # Decay step size slowly
        step_size = max(0.001, step_size * 0.999)
        
        for i in range(N):
            # Compute force from other spheres
            # Vectorized: distance from i to all j
            diffs = centers - centers[i]
            dists_sq = np.sum(diffs * diffs, axis=1)
            
            # Avoid division by zero
            # If dist is 0, force is undefined, we handle by skipping or randomizing
            # For N=100, collisions are rare after initial noise
            dists = np.sqrt(dists_sq)
            
            # Force vector: sum of (diff / dist^3)
            # F = diff / dist^3
            # We can compute this safely
            # Mask out self
            mask = dists_sq > 1e-12
            force = np.zeros(3)
            if np.any(mask):
                # dists_sq[mask] is non-zero
                # 1 / dist^3 = 1 / (dists_sq * dist)
                inv_dist_cubed = 1.0 / (dists_sq[mask] * dists[mask])
                force += np.sum(diffs[mask] * inv_dist_cubed[:, np.newaxis], axis=0)
            
            # Wall forces
            for k in range(3):
                pos = centers[i, k]
                d_wall = 0.5 - abs(pos)
                if d_wall < 0.1: # Only apply if close to wall to save time? No, always apply for smoothness
                    # Force away from wall
                    if pos < 0:
                        force[k] += 1.0 / (d_wall * d_wall)
                    else:
                        force[k] -= 1.0 / (d_wall * d_wall)
            
            # Update
            centers[i] += step_size * force
            centers[i] = np.clip(centers[i], -0.5, 0.5)
        
        # Check improvement
        current_radius = calc_radius(centers)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()
            
    return best_centers
