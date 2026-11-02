"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 26

# EVOLVE-BLOCK-START
def construct():
    best_score = -np.inf
    best_result = None
    
    # Parameters for Simulated Annealing
    T_start = 0.1
    T_end = 1e-4
    cooling_rate = 0.9995
    num_restarts = 20
    
    for restart in range(num_restarts):
        np.random.seed(restart * 12345)
        
        # Initialize with random positions
        centers = np.random.uniform(0.05, 0.95, size=(N, 2))
        radii = np.zeros(N)
        
        # Initial radii: set to max possible given current positions (greedy init)
        for i in range(N):
            dists = np.linalg.norm(centers - centers[i], axis=1)
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            # Max radius is limited by distance to others and boundaries
            # r_i <= (d_ij - r_j) / 2
            # Since we start with r_j=0, r_i <= d_ij / 2
            # Also r_i <= min(x, 1-x, y, 1-y)
            candidate_r = np.min((dists[mask]) / 2.0)
            bound_r = min(centers[i][0], 1.0 - centers[i][0], centers[i][1], 1.0 - centers[i][1])
            radii[i] = min(candidate_r, bound_r, 0.1) # Cap initial radius to prevent huge overlaps in init
            
        T = T_start
        steps = 0
        max_steps = 50000
        
        current_energy = -np.sum(radii)
        
        while T > T_end and steps < max_steps:
            # Pick a random circle
            i = np.random.randint(N)
            
            # Generate a new random position
            # We can use a Gaussian perturbation around current pos or a fully random pos.
            # Fully random is better for exploration early on, but might be too disruptive.
            # Let's use a mix: with prob 0.5, random in square; with prob 0.5, perturb current.
            if np.random.rand() < 0.5:
                new_center = np.random.uniform(0.0, 1.0, size=2)
            else:
                step_size = 0.1 * (T / T_start) + 0.01 # Adaptive step size
                new_center = centers[i] + np.random.normal(0, step_size, size=2)
                new_center = np.clip(new_center, 0.0, 1.0)
            
            # Calculate new radius for circle i at new_center
            dists = np.linalg.norm(centers - new_center, axis=1)
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            
            # Constraint from other circles: d_ij >= r_i + r_j => r_i <= d_ij - r_j
            # Note: We must ensure d_ij > r_j for valid r_i > 0
            # If d_ij <= r_j, then r_i can be 0 (or negative, which we clamp to 0)
            r_constrained_by_others = np.min(dists[mask] - radii[mask])
            
            # Constraint from boundaries
            r_constrained_by_bounds = min(new_center[0], 1.0 - new_center[0], new_center[1], 1.0 - new_center[1])
            
            new_radius = max(0.0, min(r_constrained_by_others, r_constrained_by_bounds))
            
            # Calculate energy change
            # Energy = -sum(radii)
            # Old contribution: -radii[i]
            # New contribution: -new_radius
            delta_energy = -(new_radius - radii[i])
            
            # Metropolis criterion
            if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / T):
                # Accept move
                centers[i] = new_center
                radii[i] = new_radius
                current_energy += delta_energy
                
                # Track best
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack((centers, radii)).copy()
            
            # Cool down
            T *= cooling_rate
            steps += 1
            
    return best_result
# EVOLVE-BLOCK-END
