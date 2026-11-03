"""Rewritten program: simulated annealing with repulsive potential for circle packing."""
import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the largest common radius for given centers in [-0.5, 0.5]^2."""
    # Distance between all pairs
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundaries
    min_dist_to_boundary = np.min(np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    
    return min(min_pair_dist / 2.0, min_dist_to_boundary)


def _energy(centers, r_target):
    """Compute energy: penalize overlaps and boundary violations."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Pair overlap penalty
    pair_violations = np.maximum(0, r_target - dists / 2.0)
    pair_energy = np.sum(pair_violations**2)
    
    # Boundary violation penalty
    boundary_violations = np.maximum(0, np.abs(centers) - (0.5 - r_target))
    boundary_energy = np.sum(boundary_violations**2)
    
    return pair_energy + boundary_energy


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Initialize with a grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Simulated annealing parameters
    T_start = 0.05
    T_end = 0.0001
    n_iterations = 50000
    alpha = (T_end / T_start) ** (1.0 / n_iterations)
    
    T = T_start
    current_radius = best_radius
    
    # Track best
    for i in range(n_iterations):
        if time.time() > deadline:
            break
        
        # Pick a random center to move
        idx = np.random.randint(N)
        
        # Generate a new position
        new_pos = centers[idx].copy()
        new_pos += np.random.normal(0, T, size=2)
        
        # Keep within bounds
        new_pos = np.clip(new_pos, -0.5, 0.5)
        
        # Compute new radius if we move this center
        old_center = centers[idx]
        centers[idx] = new_pos
        
        new_radius = _compute_radius(centers)
        
        # Metropolis criterion
        if new_radius > current_radius:
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = centers.copy()
        else:
            # Accept with probability exp((new - current) / T)
            delta = new_radius - current_radius
            if np.random.random() < np.exp(delta / T):
                current_radius = new_radius
            else:
                centers[idx] = old_center
        
        T *= alpha
    
    # Final local optimization: try to push circles apart
    if time.time() < deadline - 5:
        for _ in range(200):
            if time.time() > deadline:
                break
            improved = False
            for idx in range(N):
                if time.time() > deadline:
                    break
                # Try small random perturbations
                for _ in range(10):
                    old_center = best_centers[idx].copy()
                    perturbation = np.random.normal(0, 0.001, size=2)
                    best_centers[idx] = np.clip(old_center + perturbation, -0.5, 0.5)
                    new_r = _compute_radius(best_centers)
                    if new_r > best_radius:
                        best_radius = new_r
                        improved = True
                    else:
                        best_centers[idx] = old_center
            if not improved:
                break
    
    return best_centers

# EVOLVE-BLOCK-END
