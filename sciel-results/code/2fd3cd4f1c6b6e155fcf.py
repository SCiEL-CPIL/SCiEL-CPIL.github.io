"""Program for packing 200 equal circles in a unit square using Simulated Annealing."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Initialize with random points
    rng = np.random.default_rng(42)
    centers = rng.uniform(-0.5, 0.5, size=(N, 2))
    
    # Helper to calculate minimum radius
    def calc_min_radius(c):
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        # Exclude diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Boundary distances
        dx = 0.5 - np.abs(c[:, 0])
        dy = 0.5 - np.abs(c[:, 1])
        min_boundary = np.min(np.minimum(dx, dy))
        
        return 0.5 * min(min_pair_dist, min_boundary)
    
    current_min_r = calc_min_radius(centers)
    best_centers = centers.copy()
    best_min_r = current_min_r
    
    # Simulated Annealing Parameters
    T_start = 0.05
    T_end = 1e-7
    iterations = 150000
    cooling_rate = (T_end / T_start) ** (1.0 / iterations)
    
    for i in range(iterations):
        T = T_start * (cooling_rate ** i)
        if T < 1e-9:
            break
            
        # Pick a random circle to move
        idx = rng.integers(0, N)
        
        # Generate a random displacement
        step_size = 0.05 * (T / T_start)
        delta = rng.normal(0, step_size, size=2)
        
        new_centers = centers.copy()
        new_centers[idx] += delta
        
        # Clamp to bounds
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        # Calculate new min radius
        new_min_r = calc_min_radius(new_centers)
        
        # Acceptance criterion
        delta_r = new_min_r - current_min_r
        
        if delta_r > 0:
            # Improvement
            centers = new_centers
            current_min_r = new_min_r
            if current_min_r > best_min_r:
                best_min_r = current_min_r
                best_centers = centers.copy()
        else:
            # Worse, accept with probability exp(delta_r / T)
            if rng.random() < np.exp(delta_r / T):
                centers = new_centers
                current_min_r = new_min_r
                if current_min_r > best_min_r:
                    best_min_r = current_min_r
                    best_centers = centers.copy()
                    
    # Final check
    if best_min_r > calc_min_radius(centers):
         return best_centers
    return centers
# EVOLVE-BLOCK-END
