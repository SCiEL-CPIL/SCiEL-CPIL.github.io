import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the centers of 200 equal circles in a unit square [-0.5, 0.5]^2
    to maximize the common radius.
    
    Strategy:
    1. Initialize points in a regular grid (slightly perturbed).
    2. Use a physics-based relaxation (Lennard-Jones style) to maximize the minimum distance.
       - Force is repulsive if distance < current_r * 2 (overlap or close).
       - Force is attractive if distance > current_r * 2 (to keep them close/tight).
       - Boundary forces push points inward if they are too close to the edge.
    3. Adaptively increase the target radius 'r' as the minimum distance grows.
    4. Use simulated annealing-like temperature decay to escape local minima.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Initial guess for radius based on area: sqrt(N * pi / 1) ~ 25 -> 1/25 = 0.04
    # But packing density is lower. Start small to allow movement.
    current_r = 0.02
    
    # Initialize in a grid
    # Approx 15x13 grid
    rows = 15
    cols = 14
    # Adjust to get exactly N
    # 15 * 14 = 210. We need 200.
    # Let's just use a 10x20 grid? Or 14x15?
    # 14 * 15 = 210.
    # Let's use a grid that covers the square.
    points = np.random.rand(N, 2) * 1.0 - 0.5
    
    # Better init: Hexagonal grid initialization to start closer to optimum
    # This helps the physics relaxation converge faster.
    k_rows = 15
    m_cols = 14
    # Create a hex grid
    dx = 1.0 / (m_cols + 1)
    dy = dx * np.sqrt(3) / 2.0
    
    # Center the grid
    # Total width approx m_cols * dx
    # Total height approx k_rows * dy
    
    init_points = []
    for i in range(k_rows):
        y = (i - (k_rows - 1) / 2.0) * dy
        for j in range(m_cols):
            x = (j - (m_cols - 1) / 2.0) * dx
            if i % 2 != 0:
                x += dx / 2.0
            init_points.append([x, y])
    
    # We have 210 points, need 200. Remove 10 random ones from the edges? 
    # Or just take the first 200? Taking first 200 might bias towards top-left.
    # Let's shuffle and take 200.
    idx = np.random.permutation(len(init_points))
    points = np.array(init_points)[idx[:N]]
    
    # Scale points to fit in [-0.5, 0.5] roughly
    # Current extent:
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    # Normalize to fit in a box slightly smaller than 1x1 to start
    scale = 0.8 / max(max_x - min_x, max_y - min_y)
    points *= scale
    points -= points.mean(axis=0)
    
    # Physics parameters
    # We want to maximize min_dist.
    # Force F_ij = k * (r_target - dist) if dist < r_target else 0?
    # No, that's just constraint satisfaction.
    # Let's use a potential energy function:
    # E = sum_{i<j} max(0, r_target - dist_ij)^2 + sum_i max(0, r_target - (0.5 - |x_i|))^2 + ...
    # Gradient descent on this energy minimizes overlaps.
    # As we decrease E, we can increase r_target.
    
    # Simulated Annealing parameters
    T = 1e-3
    T_min = 1e-6
    alpha = 0.995
    
    best_points = points.copy()
    best_min_dist = 0.0
    
    # Precompute indices for efficiency? N=200, N^2/2 = 20k pairs. 
    # 20k operations per step is fast in Python if vectorized.
    
    for step in range(20000):
        if time.time() > deadline:
            break
            
        # Calculate distances
        # Vectorized distance matrix
        # dist[i,j] = |p_i - p_j|
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        min_dist = np.min(dist)
        
        # Update best
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_points = points.copy()
            # If we found a new best, we can try to increase the target radius slightly
            # But careful: if we increase target, forces might become unstable.
            # Let's keep target_r = min_dist * 1.05 for the force calculation
            # Actually, let's just use the current min_dist as the reference for forces.
            # We want to push points apart if they are closer than 'current_r'
            # and pull them together if they are farther? No, pulling together reduces min_dist.
            # We just want to maximize min_dist.
            # So we should push apart any pair closer than a "comfortable" distance.
            # Let comfortable_dist = min_dist * 1.1
            
        # Calculate forces
        # Force is repulsive if dist < r_repulse
        # Let r_repulse = min_dist * 1.2
        r_repulse = min_dist * 1.2
        
        forces = np.zeros_like(points)
        
        # Vectorized force calculation
        # F_ij direction = (p_i - p_j) / dist_ij
        # Magnitude = k * (r_repulse - dist_ij) if dist_ij < r_repulse else 0
        
        # Mask for pairs within r_repulse
        mask = dist < r_repulse
        # Avoid division by zero (diagonal is inf, so mask is False there anyway)
        
        # Only compute for True entries? 
        # dist is 200x200. 
        # We can compute all, then zero out.
        
        # Direction vectors
        # diff is p_i - p_j
        # Unit vector u_ij = diff / dist
        # We need to handle dist=0 (coincident points). Add small epsilon.
        dist_safe = np.where(dist > 1e-10, dist, 1.0)
        u = diff / dist_safe[:, :, np.newaxis]
        
        # Magnitude
        # mag = k * (r_repulse - dist)
        # Only for mask
        mag = np.zeros_like(dist)
        mag[mask] = (r_repulse - dist[mask])
        
        # Force on i from j: F_ij = mag * u_ij
        # Total force on i = sum_j F_ij
        # Note: F_ji = -F_ij. 
        # Summing over j for each i:
        # forces[i] = sum_j mag[i,j] * u[i,j]
        
        forces = np.sum(mag[:, :, np.newaxis] * u, axis=1)
        
        # Boundary forces
        # Push points away from walls if they are within r_wall of the wall
        # r_wall = min_dist * 1.0
        r_wall = min_dist
        
        # x boundaries: -0.5 and 0.5
        # If x < -0.5 + r_wall, push right (+x)
        # If x > 0.5 - r_wall, push left (-x)
        
        x = points[:, 0]
        y = points[:, 1]
        
        # Left wall
        dist_left = x - (-0.5)
        mask_left = dist_left < r_wall
        forces[mask_left, 0] += (r_wall - dist_left[mask_left])
        
        # Right wall
        dist_right = 0.5 - x
        mask_right = dist_right < r_wall
        forces[mask_right, 0] -= (r_wall - dist_right[mask_right])
        
        # Bottom wall
        dist_bottom = y - (-0.5)
        mask_bottom = dist_bottom < r_wall
        forces[mask_bottom, 1] += (r_wall - dist_bottom[mask_bottom])
        
        # Top wall
        dist_top = 0.5 - y
        mask_top = dist_top < r_wall
        forces[mask_top, 1] -= (r_wall - dist_top[mask_top])
        
        # Update positions
        # Step size
        # Scale forces to reasonable magnitude
        # Max force can be large if points overlap significantly.
        # Normalize forces?
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 0:
            step_size = 0.1 / max_force
        else:
            step_size = 0.1
            
        new_points = points + step_size * forces
        
        # Add random noise (Annealing)
        noise = np.random.normal(0, T, size=N * 2).reshape(N, 2)
        new_points += noise
        
        # Keep inside square (hard constraint)
        new_points[:, 0] = np.clip(new_points[:, 0], -0.5, 0.5)
        new_points[:, 1] = np.clip(new_points[:, 1], -0.5, 0.5)
        
        # Check if new configuration is valid (no coincident points, inside)
        # Calculate new min dist
        diff_new = new_points[:, np.newaxis, :] - new_points[np.newaxis, :, :]
        dist_new = np.linalg.norm(diff_new, axis=2)
        np.fill_diagonal(dist_new, np.inf)
        min_dist_new = np.min(dist_new)
        
        # Acceptance criterion
        # If min_dist_new > min_dist, accept.
        # Else accept with probability exp((min_dist_new - min_dist) / T)
        
        if min_dist_new > min_dist:
            points = new_points
        else:
            delta = min_dist_new - min_dist
            if T > 0:
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    points = new_points
        
        # Cool down
        T *= alpha
        
    # Return the best configuration found
    # Ensure the best configuration is strictly inside and valid
    # The evaluator calculates r from the returned points.
    # We should return best_points.
    
    return best_points
# EVOLVE-BLOCK-END
