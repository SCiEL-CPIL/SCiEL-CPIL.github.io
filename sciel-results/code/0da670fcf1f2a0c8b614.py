"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import random

def construct():
    """
    Construct centers of 100 equal spheres in a unit cube [-0.5, 0.5]^3
    to maximize the minimum distance between any two centers and the
    distance to the cube boundary.
    
    Uses a physics-based molecular dynamics approach:
    1. Initialize particles randomly or on a grid.
    2. Simulate repulsive forces between particles and from boundaries.
    3. Use velocity Verlet integration with damping (annealing) to find
       a stable configuration maximizing the minimum separation.
    """
    np.random.seed(42)
    random.seed(42)
    
    def compute_score(centers):
        """Compute the packing score (minimum radius)."""
        if centers.shape != (N, 3):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        
        # Compute all pairwise distances
        dist_matrix = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=-1))
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair_dist = np.min(dist_matrix)
        
        # Distance to boundary
        boundary_dists = 0.5 - np.abs(centers)
        min_boundary = np.min(boundary_dists)
        
        return min(min_pair_dist / 2.0, min_boundary)

    def simulate(initial_centers, iterations=2000, initial_temp=0.1, cooling_rate=0.999):
        """
        Run molecular dynamics simulation to maximize minimum distance.
        """
        centers = initial_centers.copy()
        velocities = np.zeros_like(centers)
        
        # Parameters for force calculation
        # We want to maximize min_dist.
        # Force is repulsive when distance < target_distance.
        # We adapt target_distance or just use a strong repulsion for close pairs.
        # A common strategy is to use a potential like 1/d^2 or (d_target - d)^2.
        # Here we use a simple repulsion: F = k * (1/d^2 - 1/d_ref^2) if d < d_ref?
        # Actually, simpler: just push apart any pair that is closer than a threshold.
        # To find the *maximum* minimum distance, we can just apply repulsive forces
        # between all pairs and let them spread out until constrained by walls and each other.
        
        # Better approach: 
        # Objective: Maximize r such that all d_ij >= 2r and d_wall >= r.
        # This is equivalent to finding the configuration where the minimum distance is maximized.
        # We can simulate particles repelling each other with a force that increases as distance decreases.
        
        dt = 0.01
        damping = 0.99
        k_repel = 10.0
        
        for it in range(iterations):
            # Temperature annealing
            temp = initial_temp * (cooling_rate ** it)
            
            # Calculate forces
            forces = np.zeros_like(centers)
            
            # Pairwise repulsion
            # Compute distance matrix
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)
            dist = np.sqrt(dist_sq + 1e-12)
            
            # Mask diagonal
            np.fill_diagonal(dist, np.inf)
            np.fill_diagonal(dist_sq, np.inf)
            
            # Force magnitude: We want to push apart particles.
            # If we just use 1/dist^2, they will fly apart until hitting walls.
            # The equilibrium will be determined by the walls and the mutual repulsion.
            # To maximize the *minimum* distance, we need a force that is very strong for close pairs.
            # F = k * (1/dist^2)
            # However, this doesn't explicitly target the minimum. It targets total energy minimization 
            # of a repulsive potential. For hard spheres, this tends to maximize the minimum distance 
            # in a confined box.
            
            # To be safe and ensure we maximize the *minimum*, we can use a force that is proportional
            # to the gradient of the minimum distance? No, that's non-differentiable.
            # Standard MD with repulsive forces works well for sphere packing.
            
            inv_dist_sq = 1.0 / dist_sq
            # Avoid division by zero
            np.fill_diagonal(inv_dist_sq, 0.0)
            
            # Force vector = k * (r_i - r_j) / dist^2
            # F_ij = k * diff_ij / dist_ij^2
            # Sum over j for each i
            # forces[i] = sum_j k * (centers[i] - centers[j]) / dist_ij^2
            
            # Efficient calculation:
            # factor = k / dist_sq
            # factor[i,i] = 0
            # forces = sum_j factor[i,j] * (centers[i] - centers[j])
            
            # Note: diff[i,j] = centers[i] - centers[j]
            # So force on i from j is k * diff[i,j] / dist_sq[i,j]
            
            force_factor = k_repel * inv_dist_sq
            # Multiply each component of diff by force_factor
            # diff is (N, N, 3)
            # force_factor is (N, N)
            # We need to sum over j (axis 1)
            
            # forces[i] = sum_j force_factor[i,j] * diff[i,j]
            # This can be done by:
            # weighted_diff = diff * force_factor[:, :, np.newaxis]
            # forces = np.sum(weighted_diff, axis=1)
            
            weighted_diff = diff * force_factor[:, :, np.newaxis]
            pair_forces = np.sum(weighted_diff, axis=1)
            
            # Boundary repulsion
            # Push particles away from walls if they are too close.
            # Let's define a "soft wall" potential.
            # If center > 0.5 - r_margin, push left.
            # We don't know r yet. We can just push if outside [-0.5, 0.5].
            # But we want them inside.
            # Force from wall: if x > 0.5, F_x = -k_wall * (x - 0.5)^2?
            # Or simpler: if x > 0.5, F_x = -k_wall.
            # Let's use a quadratic potential for smoothness: V = 0.5 * k_wall * (violation)^2
            # F = -dV/dx = -k_wall * violation
            
            k_wall = 100.0
            wall_forces = np.zeros_like(centers)
            
            # Right wall (x > 0.5)
            viol_r = np.maximum(0, centers[:, 0] - 0.5)
            wall_forces[:, 0] -= k_wall * viol_r
            # Left wall (x < -0.5)
            viol_l = np.maximum(0, -0.5 - centers[:, 0])
            wall_forces[:, 0] += k_wall * viol_l
            
            # Right wall (y > 0.5)
            viol_r = np.maximum(0, centers[:, 1] - 0.5)
            wall_forces[:, 1] -= k_wall * viol_r
            # Left wall (y < -0.5)
            viol_l = np.maximum(0, -0.5 - centers[:, 1])
            wall_forces[:, 1] += k_wall * viol_l
            
            # Right wall (z > 0.5)
            viol_r = np.maximum(0, centers[:, 2] - 0.5)
            wall_forces[:, 2] -= k_wall * viol_r
            # Left wall (z < -0.5)
            viol_l = np.maximum(0, -0.5 - centers[:, 2])
            wall_forces[:, 2] += k_wall * viol_l
            
            forces = pair_forces + wall_forces
            
            # Update velocities (Verlet-like with damping)
            # v_new = (v_old + a * dt) * damping
            # a = F / m (m=1)
            velocities = (velocities + forces * dt) * damping
            
            # Add some noise to escape local minima?
            # velocities += np.random.normal(0, temp * 0.1, size=centers.shape)
            
            # Update positions
            centers = centers + velocities * dt
            
            # Hard clip to prevent escaping bounds entirely (safety)
            centers = np.clip(centers, -0.5, 0.5)
            
        return centers

    best_centers = None
    best_score = 0.0
    
    # Strategy 1: Jittered Grid
    m = int(np.ceil(N ** (1.0 / 3.0)))
    # 5x5x5 = 125 points, take 100
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_points = []
    for z in ticks:
        for y in ticks:
            for x in ticks:
                grid_points.append([x, y, z])
    
    grid_points = np.array(grid_points)
    # Shuffle and take N
    idx = np.random.choice(len(grid_points), N, replace=False)
    initial = grid_points[idx] + np.random.uniform(-0.01, 0.01, size=(N, 3))
    initial = np.clip(initial, -0.5, 0.5)
    
    centers = simulate(initial, iterations=3000, initial_temp=0.5, cooling_rate=0.9995)
    score = compute_score(centers)
    if score > best_score:
        best_score = score
        best_centers = centers.copy()
        
    # Strategy 2: Random Initialization
    for restart in range(5):
        initial = np.random.uniform(-0.45, 0.45, size=(N, 3))
        centers = simulate(initial, iterations=3000, initial_temp=0.5, cooling_rate=0.9995)
        score = compute_score(centers)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            
    # Strategy 3: Refine the best solution with a longer run
    if best_centers is not None:
        # Use a smaller step and longer run to polish
        centers = simulate(best_centers, iterations=5000, initial_temp=0.1, cooling_rate=0.9999)
        score = compute_score(centers)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            
    # Final safety check and clip
    if best_centers is None:
        best_centers = np.random.uniform(-0.45, 0.45, size=(N, 3))
        
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

# EVOLVE-BLOCK-END
