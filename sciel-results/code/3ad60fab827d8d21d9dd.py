"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START

def _hexagonal_init(n, box_size=1.0):
    """
    Generate a hexagonal lattice of points within a square of side box_size.
    Returns an array of shape (n, 2).
    """
    # Hexagonal lattice spacing
    # Distance between rows is s * sqrt(3)/2
    # Distance between cols is s
    # We want to fit roughly sqrt(n) points.
    
    # Start with a guess for spacing
    # Area per point ~ box_size^2 / n
    # Hex area per point ~ s^2 * sqrt(3)/2
    # s ~ sqrt(2 * box_size^2 / (n * sqrt(3)))
    s = np.sqrt(2.0 * box_size**2 / (n * np.sqrt(3)))
    
    # Generate a grid larger than the box to allow clipping
    # Estimate number of rows/cols needed
    num_cols = int(np.ceil(box_size / s)) + 2
    num_rows = int(np.ceil(box_size / (s * np.sqrt(3) / 2))) + 2
    
    points = []
    for row in range(num_rows):
        for col in range(num_cols):
            x = col * s
            y = row * s * np.sqrt(3) / 2
            if row % 2 == 1:
                x += s / 2
            points.append([x, y])
            
    points = np.array(points)
    
    # Center the grid
    points -= points.mean(axis=0)
    
    # Filter points inside the box [-0.5, 0.5]
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    # If we have too many, randomly subsample
    if len(points) > n:
        indices = np.random.choice(len(points), n, replace=False)
        points = points[indices]
    elif len(points) < n:
        # Add random points to fill the gap
        num_to_add = n - len(points)
        random_points = np.random.uniform(-0.5, 0.5, size=(num_to_add, 2))
        points = np.vstack([points, random_points])
        
    # Add a small random jitter to break symmetry
    points += np.random.uniform(-0.01, 0.01, size=points.shape)
    
    return points

def _relax(points, r, max_iters=1000, step_size=0.1):
    """
    Relax points to satisfy non-overlapping constraint for radius r.
    Returns updated points and success boolean.
    """
    n = len(points)
    points = points.copy()
    
    for it in range(max_iters):
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        # Compute boundary distances
        # dist to left: x + 0.5
        # dist to right: 0.5 - x
        # dist to bottom: y + 0.5
        # dist to top: 0.5 - y
        dist_left = points[:, 0] + 0.5
        dist_right = 0.5 - points[:, 0]
        dist_bottom = points[:, 1] + 0.5
        dist_top = 0.5 - points[:, 1]
        
        # Minimum distance to boundary for each point
        min_dist_bound = np.minimum(np.minimum(dist_left, dist_right), np.minimum(dist_bottom, dist_top))
        
        # Check if all constraints are satisfied
        min_pair_dist = np.min(dist)
        min_bound_dist = np.min(min_dist_bound)
        
        if min_pair_dist >= 2 * r and min_bound_dist >= r:
            return points, True
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion
        # Only consider pairs with dist < 2r
        mask = dist < 2 * r
        # Exclude diagonal
        mask[np.diag_indices_from(mask)] = False
        
        # For each active pair, add force
        # Vectorized approach for forces is tricky with variable counts per point.
        # Looping over points might be faster for N=200 than complex vectorized logic?
        # Let's try a semi-vectorized approach.
        
        # Calculate displacement vectors for all active pairs
        # We need to sum up (p_i - p_j) * (2r - d_ij) / d_ij for all j for each i
        
        # Create a matrix of unit vectors
        # dist is N x N
        # diff is N x N x 2
        # unit_vec = diff / dist[..., np.newaxis]
        # But dist has zeros on diagonal and we masked them.
        
        # Safe division
        dist_safe = np.where(mask, dist, 1.0)
        unit_vecs = diff / dist_safe[..., np.newaxis]
        
        # Strength of repulsion: (2r - dist)
        strength = np.where(mask, 2 * r - dist, 0.0)
        
        # Force from pair (i,j) on i is strength[i,j] * unit_vec[i,j]
        # Sum over j for each i
        forces = np.sum(unit_vecs * strength[..., np.newaxis], axis=1)
        
        # Boundary forces
        # If dist_left < r, push right (positive x)
        # Force magnitude: r - dist_left
        forces[:, 0] += np.where(dist_left < r, r - dist_left, 0.0)
        forces[:, 0] -= np.where(dist_right < r, r - dist_right, 0.0)
        forces[:, 1] += np.where(dist_bottom < r, r - dist_bottom, 0.0)
        forces[:, 1] -= np.where(dist_top < r, r - dist_top, 0.0)
        
        # Update positions
        points += step_size * forces
        
        # Clip to square
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        
        # Optional: reduce step size as we approach convergence
        if it > max_iters // 2:
            step_size *= 0.95
            
    return points, False

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_points = None
    best_radius = 0.0
    
    # Number of restarts
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        points = _hexagonal_init(N)
        
        # Initial radius estimate
        # Calculate min dist of initial points
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist)
        
        dist_left = points[:, 0] + 0.5
        dist_right = 0.5 - points[:, 0]
        dist_bottom = points[:, 1] + 0.5
        dist_top = 0.5 - points[:, 1]
        min_bound = np.min(np.minimum(np.minimum(dist_left, dist_right), np.minimum(dist_bottom, dist_top)))
        
        current_r = min(min_dist / 2, min_bound) * 0.9
        
        if current_r <= 0:
            current_r = 0.01
            
        # Optimization loop
        # We try to increase r gradually
        r_increment = 0.0005
        
        while time.time() < deadline:
            # Try to relax for current r
            # Use a moderate number of iterations
            points, success = _relax(points, current_r, max_iters=200, step_size=0.5)
            
            if success:
                # Check if we can increase r
                # Calculate actual min radius
                diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
                dist = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dist, np.inf)
                min_dist = np.min(dist)
                
                dist_left = points[:, 0] + 0.5
                dist_right = 0.5 - points[:, 0]
                dist_bottom = points[:, 1] + 0.5
                dist_top = 0.5 - points[:, 1]
                min_bound = np.min(np.minimum(np.minimum(dist_left, dist_right), np.minimum(dist_bottom, dist_top)))
                
                actual_r = min(min_dist / 2, min_bound)
                
                if actual_r > best_radius:
                    best_radius = actual_r
                    best_points = points.copy()
                    
                # Increase r
                current_r += r_increment
                # Slow down increment as r gets larger
                if current_r > 0.03:
                    r_increment *= 0.9
            else:
                # Failed to relax
                # Try a smaller increment or break
                # If we can't even relax for current r, we might be stuck
                # Try to recover by relaxing for a smaller r
                # Or just break and try next restart
                break
                
    if best_points is None:
        # Fallback to grid if something went wrong
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return best_points

# EVOLVE-BLOCK-END
