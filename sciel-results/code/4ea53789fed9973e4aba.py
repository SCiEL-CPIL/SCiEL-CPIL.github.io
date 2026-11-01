import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # 1. Initialization: Hexagonal Lattice
    # We want to generate a lattice that is dense enough to have at least N points
    # inside the unit circle, then pick the best N.
    
    # Try a few different spacings to find a good initial set
    best_points = None
    best_score = 0.0
    
    # Estimated optimal radius is ~0.053. Spacing ~ 2*0.053 = 0.106.
    # Let's try spacings around that.
    for spacing in [0.10, 0.105, 0.11, 0.115, 0.12]:
        k = int(1.0 / spacing) + 3
        # Generate hexagonal points
        # x = (i + 0.5*(j%2))*spacing
        # y = j * spacing * sqrt(3)/2
        points_list = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                points_list.append([x, y])
        
        points = np.array(points_list)
        # Filter points inside unit circle (with a small margin for safety in init)
        radii = np.hypot(points[:, 0], points[:, 1])
        mask = radii <= 1.0
        points = points[mask]
        
        if len(points) < N:
            continue
            
        # Select the N points closest to the center
        # This tends to form a more regular disk shape
        idx = np.argsort(radii[mask])[:N]
        init_points = points[idx]
        
        # Score the initial configuration
        min_dist = float('inf')
        for i in range(N):
            dists = np.hypot(init_points[i, 0] - init_points[:, 0], init_points[i, 1] - init_points[:, 1])
            dists[i] = float('inf') # ignore self
            min_dist = min(min_dist, np.min(dists))
        min_dist_to_center = np.min(1.0 - np.hypot(init_points[:, 0], init_points[:, 1]))
        score = min(min_dist / 2.0, min_dist_to_center)
        
        if score > best_score:
            best_score = score
            best_points = init_points.copy()
            
    if best_points is None:
        # Fallback: random points
        best_points = np.random.rand(N, 2) * 2 - 1
        mask = np.hypot(best_points[:, 0], best_points[:, 1]) <= 1.0
        # This fallback is unlikely to be triggered with the lattice loop above
        
    # 2. Optimization: Local Search / Relaxation
    # We will use a simple force-directed method to push circles apart.
    # We maximize r by ensuring all constraints are satisfied for a growing r.
    # Alternatively, we can just iteratively increase r and fix violations.
    
    current_points = best_points
    current_r = best_score
    
    # Optimization parameters
    max_iters = 5000
    step_size = 0.001
    decay = 0.999
    
    for iter in range(max_iters):
        if time.time() > deadline:
            break
            
        # Compute pairwise distances
        # dists[i, j] = distance between i and j
        # Using broadcasting: (N, 1, 2) - (1, N, 2)
        diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Find minimum distance
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        # Distance to boundary
        radii = np.hypot(current_points[:, 0], current_points[:, 1])
        min_dist_boundary = np.min(1.0 - radii)
        
        # Current effective radius
        r = min(min_dist / 2.0, min_dist_boundary)
        
        if r > current_r:
            current_r = r
            # If we improved, we can maybe increase step size slightly or just continue
            # step_size = min(0.005, step_size * 1.01) # Optional
        
        # Determine forces
        # We want to increase the minimum distance.
        # Force 1: Repulsion between overlapping circles (dist < 2*r_target)
        # We use a target radius slightly larger than current r to drive improvement.
        # Let target_r = r * 1.001 + 1e-5
        target_r = r * 1.0005 + 0.0001
        
        forces = np.zeros_like(current_points)
        
        # Pairwise repulsion
        # Only consider pairs where dist < 2*target_r
        # For efficiency, we can just apply force to all pairs, but weight it heavily for close ones.
        # Or use a mask.
        mask_overlap = dists < 2.0 * target_r
        # Set diagonal to False
        mask_overlap[np.diag_indices_from(mask_overlap)] = False
        
        # Calculate repulsive forces
        # For each pair (i,j) with dist < 2*target_r:
        # direction = (p_i - p_j) / dist
        # magnitude = (2*target_r - dist) / 2  (push each away by half the overlap)
        
        # Vectorized calculation:
        # diff is p_i - p_j
        # We need to sum forces for each i.
        # Force on i from j is proportional to diff[i,j] * (2*target_r - dists[i,j]) if dists[i,j] < 2*target_r
        
        # To avoid division by zero, we already set diagonal to inf in dists, but mask handles it.
        # We need to be careful with dists == 0.
        safe_dists = np.where(dists < 1e-9, 1e-9, dists)
        
        # Overlap amount
        overlap = np.where(mask_overlap, 2.0 * target_r - dists, 0.0)
        
        # Force vector contribution from j to i:
        # F_ij = overlap[i,j] * diff[i,j] / safe_dists[i,j]
        # Note: diff[i,j] = p_i - p_j. This points from j to i.
        # If we want to push i away from j, we add this vector.
        
        force_matrix = overlap[:, :, np.newaxis] * diff / safe_dists[:, :, np.newaxis]
        
        # Sum forces over j for each i
        forces += np.sum(force_matrix, axis=1)
        
        # Boundary constraint
        # If radius > 1 - target_r, pull towards center.
        mask_boundary = radii > 1.0 - target_r
        # Force is towards center: -p_i / |p_i| * (radius - (1 - target_r))
        boundary_overlap = np.where(mask_boundary, radii - (1.0 - target_r), 0.0)
        # Unit vector towards center is -p_i / radius
        # Avoid division by zero
        safe_radii = np.where(radii < 1e-9, 1e-9, radii)
        boundary_forces = -current_points / safe_radii[:, np.newaxis] * boundary_overlap[:, np.newaxis]
        forces += boundary_forces
        
        # Update positions
        # Normalize forces to prevent large jumps
        force_magnitudes = np.linalg.norm(forces, axis=1, keepdims=True)
        max_force = np.max(force_magnitudes)
        if max_force > 0:
            normalized_forces = forces / max_force
        else:
            normalized_forces = forces
            
        current_points += step_size * normalized_forces
        
        # Ensure points stay inside the unit circle (hard constraint)
        # If a point goes out, project it back
        radii_new = np.hypot(current_points[:, 0], current_points[:, 1])
        out_of_bounds = radii_new > 1.0
        if np.any(out_of_bounds):
            # Scale back to radius 1.0
            # Actually, we should scale to 1.0 - epsilon to be safe? 
            # No, the score calculation will handle the margin. 
            # But if we go outside, it's invalid. 
            # Let's project to radius 0.9999 to be safe.
            current_points[out_of_bounds] *= (0.9999 / radii_new[out_of_bounds])[:, np.newaxis]
            
        # Decay step size
        step_size *= decay
        
        # Optional: Random jitter to escape local minima
        # Add small random noise
        if iter % 100 == 0:
            current_points += np.random.randn(N, 2) * 0.0001
            # Re-project if needed
            radii_new = np.hypot(current_points[:, 0], current_points[:, 1])
            out_of_bounds = radii_new > 1.0
            if np.any(out_of_bounds):
                current_points[out_of_bounds] *= (0.9999 / radii_new[out_of_bounds])[:, np.newaxis]

    # Final Score Calculation
    # Compute exact score
    diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    radii = np.hypot(current_points[:, 0], current_points[:, 1])
    min_dist_boundary = np.min(1.0 - radii)
    
    final_r = min(min_dist / 2.0, min_dist_boundary)
    
    # If the final configuration is invalid (e.g. points outside), return the best valid one found?
    # The projection ensures they are inside. 
    # But if min_dist is very small, it might be a bad local minimum.
    # We should return the best result found during the process or the final one?
    # The final one should be better if the algorithm works.
    
    return current_points
