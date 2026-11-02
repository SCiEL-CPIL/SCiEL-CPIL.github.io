import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 40s buffer for overhead/return
    
    # 1. Initialization: Hexagonal Lattice
    # The optimal radius is approx 0.053154.
    # Spacing for hex lattice is 2*r.
    # Start with a slightly larger spacing to ensure valid initial config, 
    # then let the optimizer shrink the effective spacing by moving points closer.
    
    # Estimate spacing
    # Area of unit circle is pi. Area per circle is pi * r^2.
    # Density is high.
    # Let's just use a range of spacings and pick one that gives ~N points.
    
    best_centers = None
    best_score = 0.0

    # Generate a dense hexagonal lattice
    # We want points roughly within radius 1.
    # Max number of points in a hex lattice of radius R with spacing s is approx (pi * R^2) / (sqrt(3)/2 * s^2)
    # We want N=300.
    # s^2 approx (pi * 1) / (300 * sqrt(3)/2) = pi / (150 * 1.732) = 3.14 / 259.8 = 0.0121
    # s approx 0.11
    
    s = 0.11
    
    # Generate grid
    # Range needs to cover radius 1.
    # k = int(1/s) + 2
    k = int(1.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        y = j * s * np.sqrt(3) / 2
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            if x*x + y*y <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we don't have enough points, reduce spacing
    while len(points) < N:
        s *= 0.98
        k = int(1.0 / s) + 3
        points = []
        for j in range(-k, k + 1):
            y = j * s * np.sqrt(3) / 2
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        
    # Select the N closest points to the origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    centers = points[idx[:N]].copy()
    
    # Ensure initial validity (should be by construction if s is large enough, but just in case)
    # If any overlap, we might need to jitter, but hex lattice with s>2r is safe.
    # Our s is approx 0.11, 2*r is approx 0.106. So s > 2r. It's valid.
    
    # 2. Optimization: Lloyd's Algorithm / Gradient Ascent
    
    # Precompute indices for pairwise distances to avoid repeated indexing
    # We will use a vectorized approach for the "force" calculation.
    
    step_size = 0.01
    decay = 0.995
    noise_scale = 0.001
    
    for iter in range(50000): # High upper bound, controlled by time
        if time.time() > deadline:
            break
            
        # Compute pairwise distances
        # dist[i,j] = ||c_i - c_j||
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=-1)
        # Set diagonal to inf so it's never the min
        np.fill_diagonal(dists_sq, np.inf)
        
        # Min distance to other centers
        min_dist_sq, min_idx = np.min(dists_sq, axis=1)
        
        # Distance to boundary
        r_curr = 1.0 - np.linalg.norm(centers, axis=1)
        
        # Determine the limiting factor for each point
        # The "radius" contribution of point i is min(min_dist_to_neighbor / 2, dist_to_boundary)
        # We want to maximize the global min.
        # Heuristic: Move each point away from its nearest constraint.
        
        forces = np.zeros_like(centers)
        
        # For each point, check if the limiting factor is the neighbor or the boundary
        # limit_r_i = min(sqrt(min_dist_sq[i]) / 2, r_curr[i])
        
        # We can compute the direction to move:
        # If neighbor is closer (in terms of constraint violation potential), move away from neighbor.
        # If boundary is closer, move towards center.
        
        # Let's define the "gap" for neighbor: gap_n = sqrt(min_dist_sq[i]) - 2*current_global_min? 
        # No, simpler: just move away from the nearest neighbor if it's the bottleneck, 
        # or towards center if boundary is the bottleneck.
        
        # Actually, a robust simple heuristic:
        # Force from neighbor: (c_i - c_neighbor) / dist
        # Force from boundary: -c_i / ||c_i|| (towards origin)
        
        # We need to decide which force to apply.
        # Compare sqrt(min_dist_sq[i]) / 2 vs r_curr[i]
        
        half_min_dist = np.sqrt(min_dist_sq) / 2.0
        
        # Boolean mask: True if boundary is the limiting factor (smaller radius)
        boundary_limited = r_curr < half_min_dist
        
        # Apply forces
        # For points limited by boundary: move towards origin
        forces[boundary_limited] = -centers[boundary_limited] / (np.linalg.norm(centers[boundary_limited], axis=1, keepdims=True) + 1e-9)
        
        # For points limited by neighbor: move away from neighbor
        # Neighbor index is min_idx
        # Neighbor coords: centers[min_idx]
        # Vector from neighbor to i: centers - centers[min_idx]
        # Note: min_idx is a 1D array of indices for each row.
        # centers[min_idx] works because min_idx is shape (N,) and centers is (N,2).
        
        neighbor_vec = centers - centers[min_idx]
        neighbor_dist = np.linalg.norm(neighbor_vec, axis=1, keepdims=True)
        # Avoid division by zero (shouldn't happen with diagonal inf, but safe)
        neighbor_vec = neighbor_vec / (neighbor_dist + 1e-9)
        
        forces[~boundary_limited] = neighbor_vec[~boundary_limited]
        
        # Update positions
        centers += step_size * forces
        
        # Add small noise to escape local minima
        if iter < 1000: # Initial phase
            centers += np.random.normal(0, noise_scale, size=centers.shape)
        else:
            centers += np.random.normal(0, noise_scale * 0.1, size=centers.shape)
            
        # Decay step size
        step_size *= decay
        
        # Calculate current score
        # Recalculate min dists for scoring (we already have min_dist_sq and r_curr from before update, 
        # but we need the new values. However, for speed, we can approximate or just check every 100 iters).
        # Let's check every 100 iterations to save time, or just use the pre-update values as a proxy?
        # No, the score depends on the NEW positions.
        
        if iter % 100 == 0:
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists_sq_new = np.sum(diff**2, axis=-1)
            np.fill_diagonal(dists_sq_new, np.inf)
            min_d_new = np.sqrt(np.min(dists_sq_new, axis=1)) / 2.0
            r_new = 1.0 - np.linalg.norm(centers, axis=1)
            current_score = np.min(np.minimum(min_d_new, r_new))
            
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
                
    # Final check and return
    if best_centers is None:
        return centers
        
    # One last polish: ensure validity
    # If best_centers is invalid (overlap), fall back to last valid or current
    # But Lloyd's usually maintains validity if step size is small enough.
    # To be safe, let's verify.
    
    diff = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    min_d = np.sqrt(np.min(dists_sq, axis=1)) / 2.0
    r_bound = 1.0 - np.linalg.norm(best_centers, axis=1)
    
    if np.any(min_d < 0) or np.any(r_bound < 0):
        # Invalid, return the last known valid one or the initial lattice
        return points[idx[:N]]
        
    return best_centers

# EVOLVE-BLOCK-START
# The code above is the complete implementation.
# EVOLVE-BLOCK-END
