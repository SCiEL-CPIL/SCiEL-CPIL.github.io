"""Rewritten program using a force-directed optimization approach."""
import numpy as np

N = 200

# EVOLVE-BLOCK-START
def construct():
    # 1. Initialization: Grid with jitter
    # Start with a square grid to provide a good initial structure
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random noise to break symmetry
    # The scale of noise should be small relative to the cell size
    cell_size = 1.0 / m
    noise = np.random.normal(0, cell_size * 0.1, size=centers.shape)
    centers += noise
    
    # Ensure centers are strictly inside the square initially
    # Clip to [-0.49, 0.49] to start safely
    centers = np.clip(centers, -0.49, 0.49)
    
    # Calculate initial radius
    def calc_radius(centers):
        # Min distance to walls
        wall_dists = np.min([
            np.min(centers[:, 0] + 0.5),
            np.min(0.5 - centers[:, 0]),
            np.min(centers[:, 1] + 0.5),
            np.min(0.5 - centers[:, 1])
        ])
        
        # Min distance between centers
        # Use broadcasting for pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        return min(wall_dists, min_pair_dist) / 2.0

    current_r = calc_radius(centers)
    
    # 2. Optimization Loop
    # Parameters
    num_steps = 3000
    base_lr = 0.05
    decay = 0.999
    
    for step in range(num_steps):
        lr = base_lr * (decay ** step)
        
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        # We want to push them apart if they are closer than 2*current_r
        # Using a soft repulsion force is often more stable
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, 1.0) # Avoid division by zero
        
        # Force magnitude: proportional to overlap
        # overlap = max(0, 2*current_r - dist)
        # To be efficient, we only compute for pairs that might interact
        # But for N=200, full matrix is fine.
        
        # Vectorized force calculation
        # F_ij = k * (2r - d) * (p_i - p_j) / d  if d < 2r else 0
        # Let's use a simpler potential: U = (2r - d)^2 if d < 2r
        # Gradient is -2(2r-d) * (p_i-p_j)/d
        
        target_dist = 2.0 * current_r
        overlap = np.maximum(0, target_dist - dists)
        
        # Normalize direction
        # Avoid division by zero for coincident points (shouldn't happen with noise)
        norm_dists = np.where(dists > 1e-6, dists, 1e-6)
        unit_vecs = diff / norm_dists[:, :, np.newaxis]
        
        # Force on i from j: k * overlap * unit_vec
        # We sum forces on i from all j
        # force[i] = sum_j (k * overlap[i,j] * unit_vec[i,j])
        # Note: diff is i - j, so unit_vec points from j to i. This is correct repulsion.
        
        k = 1.0
        force_matrix = k * overlap[:, :, np.newaxis] * unit_vecs
        forces += np.sum(force_matrix, axis=1)
        
        # Wall constraints
        # Push inward if close to wall
        wall_margin = 0.5 - current_r
        
        # X-axis
        # Left wall: x < -wall_margin -> push right (+x)
        left_violation = np.maximum(0, (-wall_margin) - centers[:, 0])
        forces[:, 0] += 5.0 * left_violation # Stronger force for walls to keep them in
        
        # Right wall: x > wall_margin -> push left (-x)
        right_violation = np.maximum(0, centers[:, 0] - wall_margin)
        forces[:, 0] -= 5.0 * right_violation
        
        # Y-axis
        # Bottom wall: y < -wall_margin -> push up (+y)
        bottom_violation = np.maximum(0, (-wall_margin) - centers[:, 1])
        forces[:, 1] += 5.0 * bottom_violation
        
        # Top wall: y > wall_margin -> push down (-y)
        top_violation = np.maximum(0, centers[:, 1] - wall_margin)
        forces[:, 1] -= 5.0 * top_violation
        
        # Update positions
        centers += lr * forces
        
        # Keep centers strictly inside the square to prevent invalid states
        # We clip to a slightly smaller box to ensure they don't escape
        # The wall forces should keep them in, but clipping is a safety net
        centers = np.clip(centers, -0.5, 0.5)
        
        # Update radius to the current feasible maximum
        # We set r to the actual minimum constraint distance / 2
        # This allows the circles to "grow" as they separate
        new_r = calc_radius(centers)
        # To avoid instability, we can lerp towards the new radius
        # current_r = 0.5 * current_r + 0.5 * new_r
        # Or just set it. Setting it directly makes the repulsion stronger as they separate.
        current_r = new_r
        
    return centers
# EVOLVE-BLOCK-END
