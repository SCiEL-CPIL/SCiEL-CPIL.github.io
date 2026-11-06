"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(centers):
    """Compute the maximum valid radius for the given centers."""
    if centers is None or len(centers) == 0:
        return 0.0
    
    # Check for non-finite values
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Check if centers are within the square [-0.5, 0.5]
    # If any center is outside, the radius is 0 (invalid)
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0

    # 1. Distance to walls
    # min_k (0.5 - |c_ik|)
    # c_ik is in [-0.5, 0.5], so |c_ik| <= 0.5
    # Distance to left wall: x + 0.5
    # Distance to right wall: 0.5 - x
    # Distance to bottom wall: y + 0.5
    # Distance to top wall: 0.5 - y
    min_x = np.min(centers[:, 0])
    max_x = np.max(centers[:, 0])
    min_y = np.min(centers[:, 1])
    max_y = np.max(centers[:, 1])
    
    r_wall = min(0.5 + min_x, 0.5 - max_x, 0.5 + min_y, 0.5 - max_y)
    
    # 2. Distance between centers
    # min_{i<j} |c_i - c_j| / 2
    # Compute pairwise distances
    # Using broadcasting for 200 points: 200x200 matrix, which is fine.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Set diagonal to infinity to ignore self-distances
    np.fill_diagonal(dist, np.inf)
    
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    
    return min(r_wall, r_pair)

def _generate_hex_grid(n):
    """Generate a hexagonal packing initialization."""
    # Approximate number of points per row
    # Hex packing density is pi/sqrt(12) ~ 0.9069
    # Square packing density is 0.785
    # For N=200, sqrt(200) ~ 14.14
    # Hex grid is slightly denser, so we need a slightly larger box or more points
    # Let's just generate a grid and trim/pad
    
    points = []
    y = -0.5
    row_idx = 0
    # Estimate points per row
    # In a hex grid, vertical spacing is r*sqrt(3)
    # Horizontal spacing is 2*r
    # Let's start with a reasonable estimate for r ~ 0.035
    # Vertical span 1.0 / (0.035 * 1.732) ~ 16.5 rows
    # Horizontal span 1.0 / (0.035 * 2) ~ 14.3 cols
    
    # We will generate a grid with r=0.035 and adjust
    r_init = 0.035
    dy = r_init * np.sqrt(3)
    dx = 2 * r_init
    
    # Start from bottom left, offset by r_init to stay inside
    y_start = -0.5 + r_init
    x_start = -0.5 + r_init
    
    # Number of rows
    n_rows = int(np.ceil(1.0 / dy)) + 2
    # Number of cols
    n_cols = int(np.ceil(1.0 / dx)) + 2
    
    for i in range(n_rows):
        y = y_start + i * dy
        if y > 0.5 - r_init:
            continue
        offset = 0 if i % 2 == 0 else r_init
        for j in range(n_cols):
            x = x_start + j * dx + offset
            if x <= 0.5 - r_init:
                points.append([x, y])
                
    points = np.array(points)
    if len(points) > n:
        return points[:n]
    else:
        # If not enough, pad with random points inside (should not happen with this estimate)
        # But for safety
        while len(points) < n:
            x = np.random.uniform(-0.5+r_init, 0.5-r_init)
            y = np.random.uniform(-0.5+r_init, 0.5-r_init)
            points = np.vstack([points, [[x, y]]])
        return points

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin for 120s limit
    
    # 1. Initialization
    centers = _generate_hex_grid(N)
    
    # 2. Multi-scale push-apart optimization
    # We use a simulated annealing-like schedule for the "force" strength
    # Force is repulsive if distance < 2*r_target, attractive otherwise?
    # Actually, for packing, we just want to push apart until no overlaps.
    # Then we want to maximize the minimum distance.
    
    # Strategy:
    # Iterate:
    #   1. Find all pairs closer than current_min_dist (or a target)
    #   2. Apply repulsive forces
    #   3. Clamp to boundaries
    #   4. Update
    
    # To maximize the radius, we want to increase the minimum distance.
    # A common heuristic is to iteratively increase a "target distance" and push points apart to satisfy it.
    
    # Let's start with a target radius slightly smaller than the current best
    # Current best from hex grid:
    current_r = _compute_radius(centers)
    target_r = current_r * 0.95
    
    # Main loop
    step_size = 0.01
    for i in range(10000):
        if time.time() > deadline:
            break
            
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Identify pairs that are too close (distance < 2 * target_r)
        # We want to push them apart to reach 2 * target_r
        # Force magnitude: proportional to overlap
        # F = k * (2*target_r - dist) if dist < 2*target_r
        
        # To be efficient, we only care about close pairs.
        # Threshold
        threshold = 2 * target_r
        
        # Mask of close pairs
        mask = dist < threshold
        # Exclude diagonal
        np.fill_diagonal(mask, False)
        
        # If no overlaps, we can increase target_r slightly
        if not np.any(mask):
            target_r *= 1.001
            continue
            
        # Compute forces
        # For each point i, sum of vectors (c_i - c_j) * (overlap) for all j
        # overlap = threshold - dist
        # We want to move c_i away from c_j
        
        # Vectorized approach:
        # diff is c_i - c_j
        # We need to scale diff by (threshold - dist) / dist (to get unit vector * overlap)
        # Actually, force direction is (c_i - c_j) / dist
        # Magnitude is (threshold - dist)
        # So force = (c_i - c_j) * (threshold - dist) / dist
        
        # Handle dist=0 (should not happen after init, but safe)
        # dist is 0 on diagonal, mask is False there.
        
        # Create a matrix of scaled differences
        # scaled_diff = diff * (threshold - dist) / dist
        # Where mask is True.
        
        # To avoid division by zero, replace 0 with 1 in dist for calculation, then zero out
        safe_dist = np.where(dist > 0, dist, 1.0)
        scale = np.where(mask, (threshold - dist) / safe_dist, 0.0)
        
        # Force on i is sum over j of scale[i,j] * diff[i,j]
        # diff[i,j] is c_i - c_j
        forces = np.sum(scale[:, :, np.newaxis] * diff, axis=1)
        
        # Update centers
        centers += step_size * forces
        
        # Clamp to boundaries
        # Keep centers in [-0.5, 0.5]
        # Actually, to maximize radius, we should keep them in [-0.5+r, 0.5-r]?
        # No, the radius is determined by the min distance to wall.
        # If we clamp to [-0.5, 0.5], the wall distance is naturally handled by the distance to wall.
        # However, if a point is at 0.5, its wall distance is 0.
        # We want to maximize the MIN of (pair_dist/2, wall_dist).
        # The push-apart only handles pairs.
        # We also need to push points away from walls if they are too close.
        
        # Let's add wall forces.
        # If x < -0.5 + target_r, push right.
        # If x > 0.5 - target_r, push left.
        
        wall_force = np.zeros_like(centers)
        
        # Left wall
        margin = -0.5 + target_r
        mask_left = centers[:, 0] < margin
        wall_force[mask_left, 0] += (margin - centers[mask_left, 0])
        
        # Right wall
        margin = 0.5 - target_r
        mask_right = centers[:, 0] > margin
        wall_force[mask_right, 0] += (margin - centers[mask_right, 0])
        
        # Bottom wall
        margin = -0.5 + target_r
        mask_bottom = centers[:, 1] < margin
        wall_force[mask_bottom, 1] += (margin - centers[mask_bottom, 1])
        
        # Top wall
        margin = 0.5 - target_r
        mask_top = centers[:, 1] > margin
        wall_force[mask_top, 1] += (margin - centers[mask_top, 1])
        
        centers += step_size * wall_force
        
        # Clamp to square to ensure validity
        centers = np.clip(centers, -0.5, 0.5)
        
        # Annealing: decrease step size slightly? Or keep constant?
        # Constant step size with increasing target_r is a form of "pressure" increase.
        # Let's keep step size constant for now, or decay slightly.
        step_size = 0.01 * (1.0 - 0.0001 * i)
        
    # 3. Final Polish: Bottleneck-driven optimization
    # Find the current radius
    r = _compute_radius(centers)
    
    # Identify the bottleneck
    # Is it a pair or a wall?
    # We can do a few iterations of "push the bottleneck apart"
    
    for i in range(1000):
        if time.time() > deadline:
            break
            
        # Compute radius
        r = _compute_radius(centers)
        
        # Find min pair distance
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        min_pair_dist = np.min(dist)
        min_pair_idx = np.argmin(dist)
        i_idx, j_idx = np.unravel_index(min_pair_idx, dist.shape)
        
        # Find min wall distance
        min_x = np.min(centers[:, 0])
        max_x = np.max(centers[:, 0])
        min_y = np.min(centers[:, 1])
        max_y = np.max(centers[:, 1])
        
        r_wall = min(0.5 + min_x, 0.5 - max_x, 0.5 + min_y, 0.5 - max_y)
        
        # Determine which is the bottleneck
        if min_pair_dist / 2.0 < r_wall:
            # Pair is bottleneck
            # Push i and j apart
            vec = centers[i_idx] - centers[j_idx]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                unit_vec = vec / norm
                # Move i away from j, j away from i
                # Small step
                step = 0.001
                centers[i_idx] += unit_vec * step
                centers[j_idx] -= unit_vec * step
        else:
            # Wall is bottleneck
            # Identify which wall
            # min_x corresponds to index of min_x
            idx_min_x = np.argmin(centers[:, 0])
            idx_max_x = np.argmax(centers[:, 0])
            idx_min_y = np.argmin(centers[:, 1])
            idx_max_y = np.argmax(centers[:, 1])
            
            # Check which is smallest
            d_left = 0.5 + centers[idx_min_x, 0]
            d_right = 0.5 - centers[idx_max_x, 0]
            d_bottom = 0.5 + centers[idx_min_y, 1]
            d_top = 0.5 - centers[idx_max_y, 1]
            
            min_d = min(d_left, d_right, d_bottom, d_top)
            
            if min_d == d_left:
                centers[idx_min_x, 0] += 0.001
            elif min_d == d_right:
                centers[idx_max_x, 0] -= 0.001
            elif min_d == d_bottom:
                centers[idx_min_y, 1] += 0.001
            else:
                centers[idx_max_y, 1] -= 0.001
                
        # Clamp
        centers = np.clip(centers, -0.5, 0.5)
        
    return centers

# EVOLVE-BLOCK-END
