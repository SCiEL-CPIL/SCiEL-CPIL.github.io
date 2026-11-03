"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Initialization: Hexagonal Close Packing
    # Generate a hexagonal grid of points centered at (0.5, 0.5)
    # Hex spacing s. Radius r = s / (2 * sqrt(3)) for touching circles.
    # We want to pick 26 points.
    
    # Estimate spacing. Area of square is 1. 26 circles.
    # If they were all radius r, area ~ 26 * pi * r^2 <= 1.
    # r ~ sqrt(1 / (26*pi)) ~ 0.11.
    # Hex spacing s = 2 * r * sqrt(3) ~ 0.38.
    
    # Let's generate a dense hex grid and pick the best 26.
    # Or, simpler: start with a rough hex arrangement and let physics refine it.
    
    # Create a hexagonal lattice
    # Basis vectors for hex lattice with spacing 1
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2.0])
    
    # We will generate points in a large enough box and then center/normalize
    # Let's try to generate ~50 points and pick the 26 closest to center
    # Spacing should be roughly 1/sqrt(26) ~ 0.2? No, that's for square.
    # For hex, density is higher.
    # Let's just use a grid of 8x8 points (64 points) with hex offset.
    
    points = []
    # Spacing factor
    # If we use 5x5 grid of hex cells, that's 25 points. Too few.
    # 6x6 grid is 36 points. Good.
    # Let's generate a 7x7 grid to be safe, then pick 26 closest to center.
    
    # Define hex grid points
    # x = i * 1.0 + (j % 2) * 0.5
    # y = j * sqrt(3)/2
    size = 7
    for i in range(size):
        for j in range(size):
            x = i + (j % 2) * 0.5
            y = j * np.sqrt(3)/2.0
            points.append([x, y])
            
    points = np.array(points)
    
    # Center the points
    center = np.array([0.5, 0.5])
    # We want to map this grid to [0,1]x[0,1] roughly.
    # Find the center of the generated point cloud
    cloud_center = points.mean(axis=0)
    points_centered = points - cloud_center
    
    # We need to select 26 points closest to the origin (which corresponds to 0.5, 0.5 in final space)
    dists = np.linalg.norm(points_centered, axis=1)
    indices = np.argsort(dists)
    selected_indices = indices[:N]
    init_points = points_centered[selected_indices]
    
    # Scale and translate to fit in [0,1]x[0,1]
    # The extent of the selected points
    min_x, min_y = init_points.min(axis=0)
    max_x, max_y = init_points.max(axis=0)
    span_x = max_x - min_x
    span_y = max_y - min_y
    
    # We want to map this span to something slightly less than 1 to allow for radius
    # Let's map to [0.1, 0.9] initially, physics will expand it.
    # Or better, normalize so that the max distance from center is 0.4?
    
    # Let's just scale so that the bounding box of points fits in [0.1, 0.9]
    scale = 0.8 / max(span_x, span_y)
    init_points = init_points * scale
    init_points += 0.5
    
    # Initial radii: small, will grow
    # Estimate max radius based on nearest neighbor distance
    # For now, just set a small radius
    radii = np.full(N, 0.05)
    
    positions = init_points.copy()
    
    # 2. Physics Optimization Loop
    # Variables: positions (N, 2), radii (N,)
    
    # Parameters
    dt = 0.01
    max_iter = 20000
    # We will run until deadline or convergence
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Calculate forces
        forces = np.zeros_like(positions)
        
        # Pairwise repulsion
        # dist[i,j] = distance between i and j
        # We want to push apart if dist < r_i + r_j
        
        # Vectorized distance calculation
        # diff[i,j] = pos[i] - pos[j]
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        dists_safe = np.where(dists == 0, 1e-6, dists)
        
        # Required distance
        req_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Overlap amount (positive if overlapping)
        overlap = req_dist - dists
        
        # Mask for self and non-overlapping
        # We only apply force if overlapping
        mask = overlap > 0
        np.fill_diagonal(mask, False)
        
        # Force is proportional to overlap, direction is normalized diff
        # F_ij = k * overlap * (diff / dist)
        # Sum forces on i
        
        # Normalize diff
        norm_diff = diff / dists_safe[:, :, np.newaxis]
        
        # Apply force magnitude
        # k is stiffness
        k_repel = 10.0
        force_magnitude = k_repel * overlap * mask
        # Shape (N, N, 1)
        force_vec = force_magnitude[:, :, np.newaxis] * norm_diff
        
        # Sum over j for each i
        forces += np.sum(force_vec, axis=1)
        
        # Boundary forces
        # Push towards boundaries to maximize radius?
        # Actually, to maximize radius, we want to push circles *away* from each other
        # and *towards* the boundaries.
        # If a circle is close to a boundary, its radius is limited by that boundary.
        # To increase radius, we need to move it away from the boundary? No, that decreases radius.
        # Wait.
        # Constraint: x_i >= r_i, x_i <= 1 - r_i, etc.
        # If we move a circle towards the center, it might allow larger radius if it was blocked by a wall?
        # No, usually larger circles are pushed to corners or edges to utilize space.
        
        # Strategy:
        # 1. Repel overlapping circles.
        # 2. "Grow" radii.
        #    We can update radii to be the minimum distance to any constraint.
        #    Then, to maximize the sum, we need to find a configuration where these minimums are large.
        #    This is equivalent to maximizing the minimum distance between all objects (circles and walls).
        #    
        #    So, we can just do a "MaxMin" optimization.
        #    But simple gradient on sum of radii is tricky because radius is a min function.
        #    
        #    Alternative:
        #    Keep positions fixed, calculate max possible radius for each circle given others.
        #    r_i = min( min_j (dist_ij - r_j), dist_to_wall_i )
        #    This is a fixed point iteration.
        
        # Let's stick to the force model but adjust the goal.
        # If we just repel, they will spread out.
        # Then we calculate the maximum radius they *could* have.
        # The "score" is sum of these max radii.
        # We want to maximize this score.
        # So we need a force that pushes towards configurations with higher sum of max radii.
        
        # Gradient of sum of max radii w.r.t position is complex.
        # However, if we treat the circles as having fixed radii and just repel them,
        # they will spread to fill the space.
        # Then we calculate the radii.
        # Does spreading maximize the sum of radii?
        # Generally, yes, for convex domains.
        
        # So:
        # 1. Apply repulsive forces between circles (treat them as hard spheres with current radii).
        # 2. Apply attractive forces to boundaries? No, that compresses them.
        #    We want them to be as far from each other as possible.
        #    So just repulsion is good.
        #    But they need to stay in the box.
        #    So we need a force that keeps them in the box, but allows them to push against walls.
        #    Actually, if they repel, they will naturally push against walls.
        #    We just need to ensure they don't leave the box.
        #    If a circle hits a wall, it should stop.
        
        # Let's refine the loop:
        # 1. Update positions based on repulsive forces.
        # 2. Clip positions to [0, 1].
        # 3. Calculate new radii as the minimum distance to any other circle or wall.
        # 4. Check if sum of radii improved.
        
        # Update positions
        positions += forces * dt
        
        # Keep in bounds (hard constraint)
        # Note: If we clip, we might create overlaps with walls?
        # The radius calculation will handle wall distance.
        # But if we clip center to [0,1], radius can be up to 0.5.
        # However, if center is at 0, radius is limited by x=0 wall.
        
        # We should probably keep centers in [0, 1] but the radius logic will
        # effectively push them in if they are too close to walls?
        # No, if center is at 0.1 and r=0.2, it overlaps wall.
        # My radius calculation: r_i = min(dist_to_wall, dist_to_other - r_other)
        # dist_to_wall = min(x, 1-x, y, 1-y)
        # So if x=0.1, dist_to_wall=0.1. So r_i <= 0.1.
        # This is consistent.
        
        # So we just clip positions to [0, 1].
        positions = np.clip(positions, 0.0, 1.0)
        
        # Update Radii
        # r_i = min( min_j (dist_ij - r_j), min_dist_to_wall_i )
        # This is an iterative update. We can do it in a few passes or just once per step.
        # Since N is small (26), we can do a few passes to converge radii for fixed positions.
        
        # Initialize radii for this step
        new_radii = np.zeros(N)
        
        # Wall distances
        dist_x = np.minimum(positions[:, 0], 1.0 - positions[:, 0])
        dist_y = np.minimum(positions[:, 1], 1.0 - positions[:, 1])
        wall_dist = np.minimum(dist_x, dist_y)
        
        # Start with wall dist as upper bound
        new_radii = wall_dist.copy()
        
        # Iterative relaxation for circle-circle constraints
        # r_i <= dist_ij - r_j  =>  r_i + r_j <= dist_ij
        # We want to maximize r_i subject to these.
        # This is equivalent to finding the largest spheres that fit.
        
        # Simple iteration:
        for _ in range(5): # 5 passes should be enough for N=26
            for i in range(N):
                min_r = wall_dist[i]
                for j in range(N):
                    if i == j: continue
                    d = np.linalg.norm(positions[i] - positions[j])
                    # Constraint: r_i <= d - r_j
                    # So r_i can be at most d - r_j
                    candidate = d - new_radii[j]
                    if candidate < min_r:
                        min_r = candidate
                new_radii[i] = min_r
            
        # Ensure radii are non-negative
        new_radii = np.maximum(new_radii, 0.0)
        
        # Update radii
        radii = new_radii
        
        # Check convergence / time
        if it % 100 == 0:
             # Optional: print progress
             pass

    # Final Result
    # Ensure validity one last time
    # Clip positions
    positions = np.clip(positions, 0.0, 1.0)
    
    # Recalculate radii precisely
    radii = np.zeros(N)
    for i in range(N):
        # Wall distance
        dx = np.minimum(positions[i, 0], 1.0 - positions[i, 0])
        dy = np.minimum(positions[i, 1], 1.0 - positions[i, 1])
        r_max = min(dx, dy)
        
        # Circle distances
        for j in range(N):
            if i == j: continue
            d = np.linalg.norm(positions[i] - positions[j])
            # r_i + r_j <= d
            # We assume r_j is already calculated? 
            # This sequential dependency is an issue.
            # Better to use the iterative method from above for final calc too.
            pass
            
    # Re-run the iterative radius calculation for the final positions
    wall_dist = np.minimum(np.minimum(positions[:, 0], 1.0 - positions[:, 0]), 
                           np.minimum(positions[:, 1], 1.0 - positions[:, 1]))
    final_radii = wall_dist.copy()
    
    for _ in range(10):
        for i in range(N):
            min_r = wall_dist[i]
            for j in range(N):
                if i == j: continue
                d = np.linalg.norm(positions[i] - positions[j])
                candidate = d - final_radii[j]
                if candidate < min_r:
                    min_r = candidate
            final_radii[i] = min_r
            
    final_radii = np.maximum(final_radii, 0.0)
    
    # Construct output array
    result = np.zeros((N, 3))
    result[:, 0] = positions[:, 0]
    result[:, 1] = positions[:, 1]
    result[:, 2] = final_radii
    
    return result

# EVOLVE-BLOCK-END
