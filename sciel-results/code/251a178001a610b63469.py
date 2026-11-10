"""Simulated Annealing approach for circle packing."""
import numpy as np

N = 200

def construct():
    # Initialization
    # Use a grid as a good starting point, then add noise
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    centers += np.random.normal(0, 0.01, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    def calc_radius(c):
        # Wall distances
        wall_dists = np.min([
            np.min(c[:, 0] + 0.5),
            np.min(0.5 - c[:, 0]),
            np.min(c[:, 1] + 0.5),
            np.min(0.5 - c[:, 1])
        ])
        
        # Pairwise distances
        # Efficiently compute min distance
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        return min(wall_dists, min_pair_dist) / 2.0

    current_r = calc_radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    # SA Parameters
    T_start = 0.05
    T_end = 0.001
    num_iter = 5000
    # Cooling schedule
    # T_t = T_start * (T_end / T_start) ** (t / num_iter)
    
    for i in range(num_iter):
        # Calculate temperature
        T = T_start * (T_end / T_start) ** (i / num_iter)
        
        # Generate new state: perturb a random subset of points
        # Perturb 10% of points
        n_perturb = max(1, int(0.1 * N))
        idx = np.random.choice(N, n_perturb, replace=False)
        
        new_centers = centers.copy()
        # Add Gaussian noise
        # Step size should be related to T
        step_size = T * 0.1
        new_centers[idx] += np.random.normal(0, step_size, size=(n_perturb, 2))
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_r = calc_radius(new_centers)
        
        # Acceptance criterion
        if new_r >= current_r or np.random.rand() < np.exp((new_r - current_r) / T):
            current_r = new_r
            centers = new_centers
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
                
    return best_centers
