"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Use simulated annealing to maximize the minimum distance between circles
    and from walls for 50 equal circles in a unit square [-0.5, 0.5]^2.
    
    The score is r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (0.5 - |c_ik|)).
    We want to maximize r.
    
    Approach:
    1. Initialize with a grid-like placement.
    2. Use simulated annealing with random perturbations.
    3. Objective: maximize the minimum of pairwise distances (divided by 2) and wall distances.
    4. Use a cooling schedule and accept worse moves with probability exp(delta/T).
    """
    np.random.seed(42)  # For reproducibility
    
    # Initial placement: use a hexagonal-like grid or just a square grid
    m = int(np.ceil(np.sqrt(N)))
    # Create a grid of m x m points, then take the first N
    # Better: use a more compact initial arrangement
    
    # Start with a grid
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Compute initial score
    def compute_score(centers):
        """Compute the radius r for the given centers."""
        n = len(centers)
        if n < 2:
            return 0.5
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Wall distances: min over all centers and all 4 walls
        # For each center (x, y): walls are at x=±0.5, y=±0.5
        # Distance to left wall: x + 0.5
        # Distance to right wall: 0.5 - x
        # Distance to bottom wall: y + 0.5
        # Distance to top wall: 0.5 - y
        min_wall = np.min(np.min(np.min(0.5 - np.abs(centers), axis=1)))
        
        return min(min_pair, min_wall)
    
    current_score = compute_score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    # Simulated annealing parameters
    T_start = 0.01
    T_end = 1e-6
    n_iterations = 50000
    step_size = 0.05
    
    # Cooling schedule
    alpha = (T_end / T_start) ** (1.0 / n_iterations)
    
    for it in range(n_iterations):
        T = T_start * (alpha ** it)
        
        # Pick a random center to perturb
        i = np.random.randint(N)
        old_pos = centers[i].copy()
        
        # Random perturbation
        centers[i] += np.random.normal(0, step_size, 2)
        
        # Clip to stay within bounds (with small margin)
        centers[i] = np.clip(centers[i], -0.4999, 0.4999)
        
        new_score = compute_score(centers)
        
        if new_score >= current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            # Accept with probability exp((new_score - current_score) / T)
            delta = new_score - current_score
            if np.random.rand() < np.exp(delta / T):
                current_score = new_score
            else:
                # Reject: revert
                centers[i] = old_pos
        
        # Adaptive step size
        if it % 1000 == 0:
            # Reduce step size gradually
            step_size *= 0.95
    
    return best_centers
# EVOLVE-BLOCK-END
