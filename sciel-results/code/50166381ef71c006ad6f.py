"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist

def construct():
    """
    Construct centers of 100 equal spheres in a unit cube [-0.5, 0.5]^3
    to maximize the minimum distance between any two centers and the
    distance to the cube boundary.
    
    Uses a physics-based force-directed layout with simulated annealing.
    """
    np.random.seed(42)
    
    # Initial configuration: Jittered grid
    # We want to place 100 points. A 5x4x5 grid = 100 points.
    # Or 4x5x5 = 100. Let's use 5x4x5 to have some symmetry.
    # Grid points in [-0.5, 0.5]
    # 5 points in x: -0.5 + 0.1, -0.3, -0.1, 0.1, 0.3, 0.5? No, spacing 1/4 = 0.25.
    # Let's just use a random jittered grid based on N.
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # If grid has more than N, take first N. If less, add random.
    if len(grid) >= N:
        centers = grid[:N].copy()
    else:
        centers = grid.copy()
        remaining = N - len(grid)
        random_pts = np.random.uniform(-0.45, 0.45, size=(remaining, 3))
        centers = np.vstack([centers, random_pts])
    
    # Add initial jitter
    centers += np.random.uniform(-0.02, 0.02, size=(N, 3))
    centers = np.clip(centers, -0.5, 0.5)
    
    # Simulation parameters
    T = 0.05  # Initial temperature
    cooling = 0.995
    min_T = 1e-6
    max_iters = 5000
    dt = 0.01 # Time step
    
    def compute_forces(pos):
        """
        Compute repulsive forces between particles and attractive forces to boundary.
        Force = -dU/dx. We want to maximize min distance, so we use a potential
        that is steep when distance is small.
        U_pair = 1/d^2 if d < 1, else 0? Or simply F = 1/d^3 * (pos_i - pos_j).
        Let's use a soft repulsion: F = k * (1/d^3) * (pos_i - pos_j) for d < 1.
        Boundary force: push back if close to wall.
        """
        forces = np.zeros_like(pos)
        
        # Pairwise repulsion
        dist_matrix = cdist(pos, pos)
        np.fill_diagonal(dist_matrix, 1.0) # Avoid division by zero
        
        # We want to push apart points that are too close.
        # Define a target distance? No, just repel strongly.
        # To avoid numerical instability, we cap the force.
        
        # Vectorized force calculation:
        # diff[i,j] = pos[i] - pos[j]
        # force[i] = sum_j (diff[i,j] / dist[i,j]^3) * strength
        # We only care about d < some cutoff, but for N=100, all pairs is fine.
        
        # Normalize diff by distance
        # dist_matrix has shape (N, N)
        # diff has shape (N, N, 3)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        
        # Avoid division by zero
        mask = dist_matrix < 1e-6
        dist_matrix_safe = np.where(mask, 1.0, dist_matrix)
        
        # Force magnitude ~ 1/d^2 (gradient of 1/d) or 1/d^3 (gradient of 1/d^2)
        # Let's use 1/d^3 for stronger repulsion at close range.
        # F = k * (r_i - r_j) / |r_i - r_j|^3
        # Multiply by 1/d^2 to get 1/d^5? No.
        # Potential V = 1/d. Force = -dV/dr = 1/d^2. Direction is unit vector (r_i-r_j)/d.
        # So Force vector = (r_i - r_j) / d^3.
        
        inv_dist_cubed = 1.0 / (dist_matrix_safe ** 3)
        # Set diagonal to 0
        np.fill_diagonal(inv_dist_cubed, 0.0)
        
        # Force on i from j is proportional to (pos_i - pos_j) * inv_dist_cubed[i,j]
        # Sum over j for each i
        # forces[i] = sum_j diff[i,j] * inv_dist_cubed[i,j]
        forces += np.sum(diff * inv_dist_cubed[:, :, np.newaxis], axis=1)
        
        # Boundary forces
        # If pos > 0.5 - margin, push left. If pos < -0.5 + margin, push right.
        # Let's use a linear spring force when outside a "safe zone" of 0.45
        margin = 0.45
        for k in range(3):
            # Right wall
            exceed_right = np.maximum(0, pos[:, k] - margin)
            forces[:, k] -= exceed_right * 10.0
            # Left wall
            exceed_left = np.maximum(0, -margin - pos[:, k])
            forces[:, k] -= exceed_left * 10.0
            
        return forces

    # Run simulation
    for it in range(max_iters):
        if T < min_T:
            break
            
        forces = compute_forces(centers)
        
        # Update positions
        # Add noise proportional to T
        noise = np.random.normal(0, T, size=(N, 3))
        centers += forces * dt + noise
        
        # Constrain to cube
        centers = np.clip(centers, -0.5, 0.5)
        
        # Cool down
        T *= cooling
        
    # Final Polish: Local optimization using a simple hill-climbing on the score
    # The physics simulation might leave some points slightly suboptimal.
    # We do a few iterations of moving the worst point towards a better spot.
    
    def get_score(pos):
        dist_matrix = cdist(pos, pos)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair_dist = np.min(dist_matrix)
        boundary_dists = 0.5 - np.abs(pos)
        min_boundary = np.min(boundary_dists)
        return min(min_pair_dist / 2.0, min_boundary)
    
    current_score = get_score(centers)
    
    # Simple local search: try moving each point slightly in random directions
    # This is slow but N=100 is small.
    # Better: Use scipy minimize on the final state with a tight tolerance.
    # Since we have a good starting point from physics, Nelder-Mead should converge fast.
    
    from scipy.optimize import minimize
    
    def neg_score_flat(x_flat):
        x = x_flat.reshape(N, 3)
        x = np.clip(x, -0.5, 0.5)
        return -get_score(x)
    
    # Run a few local optimizations from the physics result
    best_centers = centers.copy()
    best_score = current_score
    
    for restart in range(3):
        # Add tiny perturbation to escape local minima if stuck
        if restart > 0:
            perturbed = best_centers + np.random.normal(0, 0.001, size=(N, 3))
            perturbed = np.clip(perturbed, -0.5, 0.5)
        else:
            perturbed = best_centers
            
        res = minimize(
            neg_score_flat,
            perturbed.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        
        opt_centers = res.x.reshape(N, 3)
        opt_centers = np.clip(opt_centers, -0.5, 0.5)
        opt_score = get_score(opt_centers)
        
        if opt_score > best_score:
            best_score = opt_score
            best_centers = opt_centers
            
    return best_centers

# EVOLVE-BLOCK-END
