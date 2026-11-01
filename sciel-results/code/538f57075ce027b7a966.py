"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # 1. Initialization: Vogel Spiral
    # This provides a good initial distribution of points in a circle.
    n = N
    golden_angle = np.pi * (3 - np.sqrt(5))
    points = np.zeros((n, 2))
    for i in range(n):
        r = np.sqrt(i / n)
        theta = i * golden_angle
        points[i] = [r * np.cos(theta), r * np.sin(theta)]
    
    # 2. Molecular Dynamics Optimization
    # We simulate hard disks of radius r. We start with a small r and increase it.
    # The system settles into a configuration where disks do not overlap and stay inside the unit circle.
    # The maximum r for which this is possible is the optimal packing radius.
    
    # Parameters
    dt = 0.001
    friction = 0.9
    k_repulse = 1000.0  # Spring constant for repulsion
    k_boundary = 1000.0 # Spring constant for boundary
    max_steps_per_r = 2000
    r_start = 0.01
    r_step = 0.001
    max_r = 1.0 / np.sqrt(N) * 1.1 # Upper bound estimate
    
    velocities = np.zeros_like(points)
    current_r = r_start
    best_points = points.copy()
    best_score = 0.0
    
    # Helper to calculate score
    def calc_score(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        norms = np.linalg.norm(pts, axis=1)
        min_boundary_dist = np.min(1.0 - norms)
        return min(min_pair_dist / 2.0, min_boundary_dist)
    
    # Initial score
    current_score = calc_score(points)
    if current_score > best_score:
        best_score = current_score
        best_points = points.copy()
        
    # We will iterate over r values
    # To be efficient, we don't restart from scratch for each r, we continue from the previous state.
    
    # We need to be careful with the loop structure to fit in time.
    # Let's run a main loop that increases r and settles.
    
    total_time_budget = 110.0 # seconds
    import time
    start_time = time.time()
    
    # Start with a small r to allow the system to find a good configuration
    # The Vogel spiral has min distance approx 1/sqrt(N) * sqrt(3/pi) * 2? 
    # Let's just start small and grow.
    
    current_r = 0.02
    
    while current_r < max_r:
        if time.time() - start_time > total_time_budget:
            break
            
        # Settle the system for this r
        for step in range(max_steps_per_r):
            if time.time() - start_time > total_time_budget:
                break
                
            # Compute forces
            forces = np.zeros_like(points)
            
            # Pairwise repulsion
            # diff[i, j] = points[i] - points[j]
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # Overlap condition: dist < 2 * current_r
            # We want to push them apart.
            # Force magnitude: k * (2r - dist)
            # Direction: diff / dist
            
            # Vectorized calculation
            # Only consider pairs where dist < 2r
            mask = dists < 2 * current_r
            
            if np.any(mask):
                # Avoid division by zero (though diagonal is inf, off-diagonal could be 0 if coincident)
                # Add small epsilon to dists for division
                safe_dists = np.where(dists > 1e-6, dists, 1e-6)
                
                # Force vector = k * (2r - dist) * (diff / dist)
                # = k * (2r - dist) / dist * diff
                # Note: (2r - dist) is positive when overlapping.
                
                # We sum forces from all j on i.
                # F_i = sum_j F_ij
                # F_ij = k * (2r - d_ij) / d_ij * (p_i - p_j)
                
                # Create a matrix of scalar coefficients
                # coeff[i, j] = k * (2r - d_ij) / d_ij if mask[i,j] else 0
                coeff = np.zeros_like(dists)
                coeff[mask] = k_repulse * (2 * current_r - dists[mask]) / safe_dists[mask]
                
                # Sum over j for each i
                # F_i = sum_j coeff[i, j] * diff[i, j]
                # This is a matrix multiplication: coeff @ diff
                # But diff is (N, N, 2). coeff is (N, N).
                # We can do: forces = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
                
                forces += np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
            
            # Boundary confinement
            # If norm > 1 - r, push inward.
            norms = np.linalg.norm(points, axis=1)
            boundary_violation = norms - (1.0 - current_r)
            boundary_mask = boundary_violation > 0
            
            if np.any(boundary_mask):
                # Force = k_b * violation * (-point / norm)
                # Direction is inward: -point / norm
                # Magnitude: k_b * violation
                # F = -k_b * violation * (point / norm)
                
                # Only for masked points
                # point / norm is unit vector
                unit_vectors = points / norms[:, np.newaxis]
                # Force vector for each point
                f_boundary = -k_boundary * boundary_violation[:, np.newaxis] * unit_vectors
                # Zero out non-masked
                f_boundary[~boundary_mask] = 0
                forces += f_boundary
            
            # Update velocities and positions
            velocities = velocities * friction + forces * dt
            points += velocities * dt
            
            # Optional: Dampen velocities more aggressively to help settling
            # velocities *= 0.99
            
        # After settling, check score
        current_score = calc_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
        # Check if we can increase r
        # If the minimum distance is significantly larger than 2*current_r, we can increase r.
        # If the boundary distance is significantly larger than 0, we can increase r.
        # Basically, if the system is "loose", increase r.
        
        # Calculate actual min distance and boundary distance
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        norms = np.linalg.norm(points, axis=1)
        min_boundary_dist = np.min(1.0 - norms)
        
        # If min_pair_dist > 2 * current_r + epsilon AND min_boundary_dist > epsilon, we can grow.
        # We want to grow until the constraints are tight.
        slack_pair = min_pair_dist / 2.0 - current_r
        slack_boundary = min_boundary_dist
        
        min_slack = min(slack_pair, slack_boundary)
        
        if min_slack > 0.005:
            # There is room to grow
            current_r += r_step
        else:
            # Tight, try a smaller step or stop
            current_r += r_step * 0.1
            if current_r > max_r:
                break
            # If we've been stuck for a while, we might break, but let's just slow down growth
            # For simplicity, just continue with smaller steps.
            # To avoid infinite loop on stuck states, we can check if r increased very little
            # But the time limit protects us.

    return best_points
# EVOLVE-BLOCK-END
