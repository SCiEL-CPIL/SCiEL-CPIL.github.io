"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START
def construct():
    # 1. Hexagonal Initialization
    # Generate a hexagonal grid
    # Spacing dx = 1, dy = sqrt(3)/2
    # We want to fit as many as possible, then pick the best 200.
    
    # Create a grid large enough
    # Approximate number of points in a hexagonal cluster of radius R
    # We need 200 points.
    # Let's generate a grid over [-1, 1] x [-1, 1] to be safe, then filter.
    
    # Hexagonal lattice points
    # x = i + 0.5 * (j % 2)
    # y = j * sqrt(3) / 2
    # Scale to fit in [-0.5, 0.5]
    
    # Let's estimate the required scale.
    # If we have a cluster of ~200 points, the "radius" in grid units is ~ sqrt(200) ~ 14.
    # So we need a grid of size ~30x30.
    
    i_max = 20
    j_max = 20
    
    points = []
    for j in range(-j_max, j_max + 1):
        for i in range(-i_max, i_max + 1):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2.0
            points.append((x, y))
            
    points = np.array(points)
    
    # Normalize points to fit in [-0.5, 0.5]
    # Find the extent of the points
    # We want to scale such that the bounding box fits.
    # But we also want to select the 200 closest to the center.
    
    # First, normalize the raw grid to have a reasonable scale.
    # Let's assume the "unit" of the grid corresponds to the circle diameter roughly.
    # For hexagonal packing, distance between neighbors is 2r.
    # So if we set neighbor distance to 1, then r=0.5.
    # Then we scale the whole thing to fit in the square.
    
    # Current neighbor distance in (x,y) above:
    # dx=1, dy=sqrt(3)/2. Distance = 1.
    # So r=0.5 in this local coordinate system.
    
    # Select 200 points closest to (0,0)
    dists_to_center = np.sum(points**2, axis=1)
    idx = np.argsort(dists_to_center)[:N]
    selected_points = points[idx]
    
    # Now scale selected_points to fit in [-0.5, 0.5]
    # Find the max extent
    max_x = np.max(np.abs(selected_points[:, 0]))
    max_y = np.max(np.abs(selected_points[:, 1]))
    
    # We need to scale such that max_x * scale <= 0.5 and max_y * scale <= 0.5
    # This scaling is conservative (fits the bounding box).
    # However, the circles might stick out if the bounding box is tight but the shape is irregular.
    # Actually, if we scale to fit the bounding box, the circles are guaranteed to be inside
    # ONLY if the bounding box of the *centers* is within [-0.5+r, 0.5-r].
    # Wait. The centers must be in [-0.5+r, 0.5-r].
    # So the extent of centers must be <= 1 - 2r.
    
    # Let's do an iterative scaling or just pick a scale factor that works.
    # Let s be the scale factor.
    # Centers are selected_points * s.
    # Min distance between centers is 1 * s.
    # So radius r = s / 2.
    # Constraint: max(|x|, |y|) * s + r <= 0.5
    # max_extent * s + s/2 <= 0.5
    # s * (max_extent + 0.5) <= 0.5
    # s <= 0.5 / (max_extent + 0.5)
    
    max_extent = max(max_x, max_y)
    s = 0.5 / (max_extent + 0.5)
    
    centers = selected_points * s
    
    # Check validity
    # r = s / 2
    # Min pair dist should be >= 2r = s.
    # Min wall dist should be >= r = s/2.
    
    # Now, perform local optimization
    start_time = time.time()
    timeout = 110.0 # Keep 10s margin
    
    best_r = 0.0
    # Compute initial r
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    # dist to wall = 0.5 - max(|x|, |y|) ? No.
    # dist to vertical wall = 0.5 - |x|
    # dist to horizontal wall = 0.5 - |y|
    # min wall dist for point i = min(0.5 - |x_i|, 0.5 - |y_i|)
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    current_r = min(min_pair_dist / 2.0, min_wall_dist)
    best_r = current_r
    best_centers = centers.copy()
    
    # Local Search: Push algorithm
    # We will try to increase the minimum distance.
    
    # Pre-compute indices for efficiency? No, N=200 is small.
    
    step_size = 1e-4
    stall_count = 0
    
    while time.time() - start_time < timeout:
        # Recompute distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        
        wall_dists = np.min(0.5 - np.abs(centers), axis=1)
        min_wall_dist = np.min(wall_dists)
        min_wall_idx = np.argmin(wall_dists)
        
        # Determine the limiting factor
        if min_pair_dist / 2.0 < min_wall_dist:
            # Limit is pair
            i, j = min_pair_idx
            # Get vector
            v = centers[j] - centers[i]
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                v = v / norm
            else:
                v = np.array([1.0, 0.0]) # Fallback
            
            # Move i and j apart
            old_i = centers[i].copy()
            old_j = centers[j].copy()
            
            centers[i] -= step_size * v
            centers[j] += step_size * v
            
            # Clip to bounds
            centers[i] = np.clip(centers[i], -0.5, 0.5)
            centers[j] = np.clip(centers[j], -0.5, 0.5)
            
            # Check if it improved
            # Recalculate local min? Or global?
            # Global is expensive. Let's just check the specific pair and walls for i,j.
            # But moving i,j might hurt other pairs.
            # To be safe, we can do a quick global check every few steps, or just accept if local improves and hope global doesn't drop too much.
            # Better: Accept only if the *new* min distance (global) is >= old min distance.
            # This is expensive.
            
            # Alternative: Use a "try and revert" strategy with a fast local check.
            # If local check passes, do a full check. If full check fails, revert.
            
            # Fast local check:
            # New dist i-j
            new_dij = np.linalg.norm(centers[j] - centers[i])
            # New wall dists for i, j
            new_wi = np.min(0.5 - np.abs(centers[i]))
            new_wj = np.min(0.5 - np.abs(centers[j]))
            
            # If these are still >= current limiting factor, it's likely safe.
            # But it might violate another pair.
            
            # Let's do the full check. N=200, 1000 iterations = 1000 * 40000 ops = 40M ops.
            # This is fast in NumPy.
            
            diff2 = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists2 = np.linalg.norm(diff2, axis=2)
            np.fill_diagonal(dists2, np.inf)
            min_pair_dist2 = np.min(dists2)
            
            wall_dists2 = np.min(0.5 - np.abs(centers), axis=1)
            min_wall_dist2 = np.min(wall_dists2)
            
            new_r = min(min_pair_dist2 / 2.0, min_wall_dist2)
            
            if new_r > current_r:
                current_r = new_r
                stall_count = 0
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            else:
                # Revert
                centers[i] = old_i
                centers[j] = old_j
                stall_count += 1
                # If stalled, try a different move or random jitter
                if stall_count > 100:
                    # Random jitter
                    idx_rand = np.random.randint(N)
                    old_rand = centers[idx_rand].copy()
                    centers[idx_rand] += np.random.uniform(-step_size, step_size, 2)
                    centers[idx_rand] = np.clip(centers[idx_rand], -0.5, 0.5)
                    
                    diff3 = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
                    dists3 = np.linalg.norm(diff3, axis=2)
                    np.fill_diagonal(dists3, np.inf)
                    min_pair_dist3 = np.min(dists3)
                    wall_dists3 = np.min(0.5 - np.abs(centers), axis=1)
                    min_wall_dist3 = np.min(wall_dists3)
                    new_r3 = min(min_pair_dist3 / 2.0, min_wall_dist3)
                    
                    if new_r3 > current_r:
                        current_r = new_r3
                        stall_count = 0
                        if current_r > best_r:
                            best_r = current_r
                            best_centers = centers.copy()
                    else:
                        centers[idx_rand] = old_rand
                        # If still stalled, break or reduce step size?
                        # Just break to save time if no progress in 500 random moves
                        if stall_count > 500:
                            break
        else:
            # Limit is wall
            i = min_wall_idx
            # Determine which wall
            x, y = centers[i]
            if 0.5 - abs(x) < 0.5 - abs(y):
                # Vertical wall
                if x > 0:
                    v = np.array([-1.0, 0.0]) # Move left
                else:
                    v = np.array([1.0, 0.0]) # Move right
            else:
                # Horizontal wall
                if y > 0:
                    v = np.array([0.0, -1.0]) # Move down
                else:
                    v = np.array([0.0, 1.0]) # Move up
            
            old_i = centers[i].copy()
            centers[i] += step_size * v
            centers[i] = np.clip(centers[i], -0.5, 0.5)
            
            # Full check
            diff2 = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists2 = np.linalg.norm(diff2, axis=2)
            np.fill_diagonal(dists2, np.inf)
            min_pair_dist2 = np.min(dists2)
            
            wall_dists2 = np.min(0.5 - np.abs(centers), axis=1)
            min_wall_dist2 = np.min(wall_dists2)
            
            new_r = min(min_pair_dist2 / 2.0, min_wall_dist2)
            
            if new_r > current_r:
                current_r = new_r
                stall_count = 0
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            else:
                centers[i] = old_i
                stall_count += 1
                if stall_count > 500:
                    break

    return best_centers
# EVOLVE-BLOCK-END
