import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Initialization: Hexagonal lattice
    # Approximate spacing for 300 circles in unit circle
    # Area ~ pi, circle area ~ pi*r^2, 300*pi*r^2 < pi => r < 1/sqrt(300) ~ 0.057
    # Hex packing density ~ 0.906, so effective r ~ 0.054
    # Spacing d = 2r ~ 0.108
    # Let's start with a slightly larger spacing to ensure no overlap, then relax.
    r_init = 0.05
    d_init = 2 * r_init
    
    # Generate hex lattice points
    points = []
    # Range of i, j to cover unit circle
    # Max radius 1, so range approx 1/d_init
    range_val = int(1.0 / d_init) + 2
    for j in range(-range_val, range_val + 1):
        for i in range(-range_val, range_val + 1):
            x = (i + 0.5 * (j % 2)) * d_init
            y = j * d_init * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) < 1.0 - r_init:
                points.append([x, y])
                
    points = np.array(points)
    
    # If we have more than N, select the N closest to center? Or random subset?
    # Better: Select N points that are well-distributed.
    # Simple heuristic: Sort by distance from center, take first N? 
    # This might cluster them in the center. 
    # Better: Take a random subset or use a greedy selection.
    # For simplicity and robustness, let's just take the first N if we have enough, 
    # but ensure they are spread out. 
    # Actually, the hex lattice is regular. Taking the ones closest to center is fine 
    # because the relaxation will spread them out.
    if len(points) >= N:
        # Sort by radius to pick central ones
        radii = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radii)[:N]
        points = points[idx]
    else:
        # If not enough, add random points (should not happen with this d_init)
        while len(points) < N:
            angle = np.random.uniform(0, 2*np.pi)
            rad = np.random.uniform(0, 1-r_init)
            points = np.vstack([points, [rad*np.cos(angle), rad*np.sin(angle)]])
            
    # 2. Optimization Loop
    # We want to maximize r = min(min_pair_dist, min_boundary_dist)
    
    # Parameters for relaxation
    step_size = 0.01
    decay = 0.995
    epsilon = 0.01 # Threshold for "active" constraints relative to current r
    
    current_r = 0.0
    best_points = points.copy()
    best_r = 0.0
    
    # Initial score
    def calc_r(pts):
        # Pairwise distances
        # (N, N) matrix
        dists = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :]) ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # Boundary distances
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_bound = np.min(1.0 - radii)
        
        return min(min_pair, min_bound)
        
    best_r = calc_r(points)
    
    while time.time() < deadline:
        # Calculate distances
        dists = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.hypot(points[:, 0], points[:, 1])
        bound_dists = 1.0 - radii
        
        current_r = min(np.min(dists), np.min(bound_dists))
        
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
            
        # Identify active constraints
        # Pairs with distance close to min
        active_pairs = dists <= (current_r + epsilon * max(1.0, current_r))
        # Points with boundary distance close to min
        active_points = bound_dists <= (current_r + epsilon * max(1.0, current_r))
        
        # Calculate forces
        forces = np.zeros_like(points)
        
        # Pair forces: repulsion
        # For each active pair (i,j), push i away from j and j away from i
        # We can sum the unit vectors
        # To vectorize: 
        # diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        # norm = dists
        # unit_diff = diff / norm (where norm > 0)
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        # Avoid division by zero
        norm = dists.copy()
        norm[norm == 0] = 1e-10
        unit_diff = diff / norm[:, :, np.newaxis]
        
        # Weight by active mask
        # active_pairs is (N, N) boolean
        # We want sum over j for each i of unit_diff[i,j] * active_pairs[i,j]
        # Since active_pairs is symmetric, and unit_diff[i,j] = -unit_diff[j,i],
        # the sum of forces will naturally balance if we just sum.
        # Force on i from j is unit_diff[i,j] if active.
        # Sum over j.
        pair_forces = np.sum(unit_diff * active_pairs[:, :, np.newaxis], axis=1)
        forces += pair_forces
        
        # Boundary forces: attraction to center
        # For active points, force is -points / |points|
        # unit_radial = points / radii
        radii_safe = radii.copy()
        radii_safe[radii_safe == 0] = 1e-10
        unit_radial = points / radii_safe[:, np.newaxis]
        
        # Force is inward: -unit_radial
        bound_forces = -unit_radial * active_points[:, np.newaxis]
        forces += bound_forces
        
        # Update points
        points = points + step_size * forces
        
        # Optional: Clip to unit circle to prevent escape
        # If a point goes outside, pull it back
        new_radii = np.hypot(points[:, 0], points[:, 1])
        outside = new_radii > 1.0
        if np.any(outside):
            # Scale back to radius 1 - small_delta
            target_rad = 1.0 - 1e-6
            scale = target_rad / new_radii[outside]
            points[outside] *= scale[:, np.newaxis]
            
        # Decay step size
        step_size *= decay
        
        # Early stop if step size is very small
        if step_size < 1e-6:
            break
            
    # Final check
    final_r = calc_r(best_points)
    if final_r < best_r:
        # Should not happen, but just in case
        return best_points
        
    return best_points
# EVOLVE-BLOCK-END
