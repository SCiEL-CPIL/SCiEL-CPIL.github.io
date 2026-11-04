import numpy as np

N = 26

# EVOLVE-BLOCK-START
def construct():
    # Set random seed for reproducibility in local testing, 
    # but in competitive settings, different seeds might be used.
    # Here we use a fixed seed to ensure the solution is stable and valid.
    rng = np.random.default_rng(42)
    
    # Initialization: Random positions and small radii
    # Start with a small radius to avoid massive initial overlaps
    initial_radius = 0.05
    x = rng.uniform(initial_radius, 1 - initial_radius, N)
    y = rng.uniform(initial_radius, 1 - initial_radius, N)
    r = np.full(N, initial_radius)
    
    # Optimization parameters
    num_iterations = 3000
    learning_rate_pos = 0.05
    learning_rate_rad = 0.001
    decay_rate = 0.999
    
    # Preallocate arrays for forces
    fx = np.zeros(N)
    fy = np.zeros(N)
    
    for step in range(num_iterations):
        # 1. Calculate pairwise distances and overlaps
        # We want to push circles apart if they overlap.
        # dist[i, j] is distance between i and j
        # min_dist[i, j] is r[i] + r[j]
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        
        dist = np.sqrt(dx**2 + dy**2)
        
        # Avoid division by zero for i==j
        # We only care about i != j
        np.fill_diagonal(dist, np.inf)
        
        min_dist = r[:, None] + r[None, :]
        
        # Overlap amount: positive if overlapping
        overlap = min_dist - dist
        # Only consider positive overlaps
        overlap[overlap < 0] = 0
        
        # Force direction: unit vector from j to i
        # If dist is very small, randomize direction to break symmetry
        small_dist_mask = dist < 1e-6
        if np.any(small_dist_mask):
            # Add small random noise to positions if they are exactly on top of each other
            x[small_dist_mask.any(axis=1)] += rng.normal(0, 1e-6, size=N)
            y[small_dist_mask.any(axis=1)] += rng.normal(0, 1e-6, size=N)
            # Recalculate dist for safety, though noise should suffice
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, np.inf)
            overlap = min_dist - dist
            overlap[overlap < 0] = 0
            
        # Unit vectors
        # dist is now > 0 for i != j
        inv_dist = 1.0 / dist
        inv_dist[dist == 0] = 0 # Safety
        
        ux = dx * inv_dist
        uy = dy * inv_dist
        
        # Repulsive force is proportional to overlap
        # Sum forces from all j for each i
        # fx[i] = sum_j overlap[i,j] * ux[i,j]
        fx = np.sum(overlap * ux, axis=1)
        fy = np.sum(overlap * uy, axis=1)
        
        # 2. Boundary Forces
        # Push circles back if they are outside or too close to boundary
        # Left boundary: x < r -> force right (+)
        # Right boundary: x > 1-r -> force left (-)
        # Top/Bottom similar for y
        
        # We use a linear penalty for being out of bounds or too close
        # Actually, we should prevent them from going out.
        # If x < r, we need to push x up.
        # Let's define boundary violation.
        
        # Left
        viol_left_x = np.maximum(0, r - x)
        # Right
        viol_right_x = np.maximum(0, x - (1 - r))
        # Bottom
        viol_bottom_y = np.maximum(0, r - y)
        # Top
        viol_top_y = np.maximum(0, y - (1 - r))
        
        # Add boundary forces to fx, fy
        # Boundary forces should be strong
        boundary_strength = 10.0
        fx += boundary_strength * (viol_left_x - viol_right_x)
        fy += boundary_strength * (viol_bottom_y - viol_top_y)
        
        # 3. Update Positions
        # Use a decaying learning rate for stability
        current_lr_pos = learning_rate_pos * (decay_rate ** step)
        
        # Normalize forces? The overlap magnitude varies.
        # A simple gradient step is usually sufficient with small lr.
        x += current_lr_pos * fx
        y += current_lr_pos * fy
        
        # 4. Update Radii
        # Strategy: Try to increase radius for all circles.
        # If a circle is "free" (not causing overlaps), it can grow.
        # If it's causing overlaps, the repulsion forces pushed neighbors away, 
        # effectively clearing space.
        # We can simply increase all radii by a small amount.
        # The repulsion forces will handle the separation.
        # However, if we just increase radii, the overlaps will increase, 
        # causing larger repulsive forces in the next step.
        # This acts as a pressure that expands the packing.
        
        current_lr_rad = learning_rate_rad * (decay_rate ** step)
        
        # Increase radii
        r += current_lr_rad
        
        # Optional: If a circle is stuck against a wall, it might not be able to grow 
        # as much as one in the center. But the global pressure helps.
        
        # Clip radii to reasonable max to prevent explosion
        r = np.minimum(r, 0.5)
        
    # Final cleanup: Ensure strict validity
    # Sometimes numerical errors leave tiny overlaps.
    # We can do a final pass to shrink circles slightly if they overlap.
    for _ in range(100):
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        min_dist = r[:, None] + r[None, :]
        
        # Check max overlap
        max_overlap = np.max(min_dist - dist)
        if max_overlap < 1e-6:
            break
        
        # Shrink all circles slightly to resolve overlaps
        shrink_factor = 1.0 - max_overlap * 0.5
        r *= shrink_factor
        
        # Also ensure they are inside the box
        # If r is too big for the box, shrink further
        max_r_allowed = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        r = np.minimum(r, max_r_allowed)
        
    return np.column_stack((x, y, r))
# EVOLVE-BLOCK-END
