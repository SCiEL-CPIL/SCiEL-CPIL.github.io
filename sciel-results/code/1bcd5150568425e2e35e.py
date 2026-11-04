import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin from 120s
    
    # 1. Initialization: Phyllotaxis spiral
    # This provides a very dense, uniform initial distribution
    indices = np.arange(N)
    r_init = np.sqrt((indices + 0.5) / N)
    theta_init = indices * np.pi * (3.0 - np.sqrt(5.0))  # Golden angle
    x = r_init * np.cos(theta_init)
    y = r_init * np.sin(theta_init)
    centers = np.column_stack((x, y))
    
    # Helper to check validity and compute min radius
    def compute_radius(centers):
        # Boundary constraint
        dist_to_center = np.linalg.norm(centers, axis=1)
        min_boundary = 1.0 - np.max(dist_to_center)
        
        # Pairwise distances
        # N=300, 300x300 matrix is 90k elements, very fast
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        r = min(min_boundary, min_dist / 2.0)
        return r

    # Initial radius estimate
    current_r = compute_radius(centers)
    
    # We want to maximize r.
    # Strategy: Iteratively increase r and relax the configuration.
    # If relaxation fails, we might need to restart or decrease r.
    
    # Let's use a force-directed relaxation for a fixed target radius.
    # If the system can reach zero overlap for target_r, it's valid.
    
    def relax(centers, target_r, max_steps, alpha_start):
        """
        Attempts to arrange centers so they don't overlap with radius target_r.
        Returns (centers, success_bool, final_overlap_score)
        """
        centers = centers.copy()
        alpha = alpha_start
        
        for step in range(max_steps):
            if time.time() > deadline - 10.0:
                break
                
            # Compute forces
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            
            # Avoid division by zero
            np.fill_diagonal(dists, 1.0)
            
            # Overlap condition: dist < 2 * target_r
            # Force magnitude proportional to overlap
            overlap = np.maximum(0.0, 2.0 * target_r - dists)
            
            # Unit vectors
            # Handle dist=0 case (coincident points) by adding small random jitter or skipping
            # For simplicity, if dist is very small, we add a small random force
            unit_vecs = diff / dists[:, :, np.newaxis]
            
            # Force from neighbors: sum of unit_vecs * overlap
            # F_i = sum_j (overlap_ij * unit_vec_ij)
            # Note: unit_vec_ij points from j to i? 
            # diff = c_i - c_j. So unit_vec points from j to i.
            # If c_i and c_j overlap, we want to push c_i away from c_j.
            # So force on i is in direction (c_i - c_j). Correct.
            
            forces = np.sum(overlap[:, :, np.newaxis] * unit_vecs, axis=1)
            
            # Boundary force
            dist_to_center = np.linalg.norm(centers, axis=1)
            boundary_overlap = np.maximum(0.0, dist_to_center - (1.0 - target_r))
            
            # Force towards center
            # Direction is -c_i / |c_i|
            # Magnitude is boundary_overlap
            center_dir = -centers / np.maximum(dist_to_center, 1e-9)[:, np.newaxis]
            boundary_forces = boundary_overlap[:, np.newaxis] * center_dir
            
            total_forces = forces + boundary_forces
            
            # Update positions
            centers += alpha * total_forces
            
            # Cool down
            alpha *= 0.99
            
            # Check for convergence
            max_overlap = np.max(overlap)
            max_boundary_overlap = np.max(boundary_overlap)
            if max_overlap < 1e-6 and max_boundary_overlap < 1e-6:
                return centers, True, 0.0
        
        # Final check
        dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        dist_to_center = np.linalg.norm(centers, axis=1)
        min_boundary = 1.0 - np.max(dist_to_center)
        
        final_r = min(min_boundary, min_dist / 2.0)
        success = (final_r >= target_r - 1e-7)
        # Score: sum of overlaps (lower is better)
        overlap = np.maximum(0.0, 2.0 * target_r - dists)
        boundary_overlap = np.maximum(0.0, dist_to_center - (1.0 - target_r))
        score = np.sum(overlap) + np.sum(boundary_overlap)
        
        return centers, success, score

    # Optimization Loop
    # Start with a conservative radius
    r_current = 0.04
    best_centers = centers.copy()
    best_r = current_r
    
    # If initial is better, use it
    if current_r > r_current:
        r_current = current_r
        best_r = current_r
        best_centers = centers.copy()
        
    # We will try to increase r in small steps
    # If we fail to relax for a new r, we stop or try a different strategy.
    # Given the time limit, we should be conservative with steps.
    
    step_size = 0.001
    max_relax_steps = 2000
    
    # Pre-calculate a "good" alpha based on scale
    # Forces are roughly O(1) if overlap is O(1).
    # Alpha should be small enough to not oscillate.
    
    while time.time() < deadline - 20.0:
        # Try to increase r
        target_r = r_current + step_size
        
        # Relax
        new_centers, success, score = relax(best_centers, target_r, max_relax_steps, alpha_start=0.05)
        
        if success:
            r_current = target_r
            best_centers = new_centers
            best_r = r_current
            # Maybe increase step size if we are doing well?
            # step_size *= 1.1 
        else:
            # Failed to achieve target_r
            # Try a smaller step?
            # Or just stop?
            # Let's try to see if we can get closer by binary search between last good and current?
            # Or just break, as we are likely at a local optimum for this configuration.
            
            # Let's try one more time with a slightly lower target to see if we can polish
            # Actually, if we failed, we should probably stop increasing.
            # But we might be able to improve the current configuration at the *current* r_current
            # to see if it allows for a slightly larger r later?
            # For simplicity, break if we fail twice in a row?
            
            # Let's try to refine the current best_r configuration
            # Relax at current r_current with more steps?
            # If it's already valid, relaxing further might not help increase r immediately.
            
            # Let's break out of the loop to save time for a final polish if needed.
            break

    # Final Polish:
    # The current best_centers should be valid for best_r.
    # We can try to compute the exact max r for this configuration.
    final_r = compute_radius(best_centers)
    
    # If final_r is significantly larger than best_r (due to slack in the relaxation), update it.
    if final_r > best_r:
        best_r = final_r
    
    return best_centers
