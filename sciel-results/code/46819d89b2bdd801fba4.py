"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.spatial.distance import cdist

# Deadline guard: evaluator kills after 150s, keep safety margin
_DEADLINE = time.time() + 140.0


def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check inside square [-0.5, 0.5]^2
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    # Min distance to walls
    wall_min = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_min)
    # Min pairwise distance
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    return min(min_wall, min_pair)


def _initial_grid(n):
    """Generate a square grid of n points in [-0.5, 0.5]^2."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]


def _initial_random(n, seed=0):
    """Generate random points in [-0.5, 0.5]^2."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5, 0.5, size=(n, 2))


def _repulsion_energy(centers, power=4):
    """
    Energy function: sum of repulsive forces between close pairs + wall penalties.
    We want to maximize the minimum distance, so we minimize a potential that
    penalizes close pairs and wall violations.
    """
    centers = np.asarray(centers, dtype=np.float64)
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    
    # Pairwise repulsion: use inverse power law for close pairs
    # Only consider pairs within a cutoff for efficiency
    cutoff = 0.15
    mask = d < cutoff
    if np.any(mask):
        pair_d = d[mask]
        pair_energy = np.sum(1.0 / (pair_d ** power))
    else:
        pair_energy = 0.0
    
    # Wall penalty: push centers away from walls
    wall_dist = 0.5 - np.max(np.abs(centers), axis=1)
    wall_penalty = np.sum(1.0 / np.maximum(wall_dist, 0.01) ** power)
    
    return pair_energy + 10.0 * wall_penalty


def _gradient_step(centers, power=4, step_size=0.001, max_iter=5000):
    """
    Gradient descent on the repulsion energy to improve packing.
    Returns improved centers.
    """
    centers = np.array(centers, dtype=np.float64)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE:
            break
        
        d = cdist(centers, centers)
        np.fill_diagonal(d, np.inf)
        
        # Compute forces from pairwise repulsion
        cutoff = 0.2
        mask = (d < cutoff) & (d > 1e-10)
        if not np.any(mask):
            break
        
        # Force on each center from all others
        forces = np.zeros_like(centers)
        # For each pair i,j within cutoff, add force
        idx_i, idx_j = np.where(mask)
        for i, j in zip(idx_i, idx_j):
            diff = centers[i] - centers[j]
            dist = d[i, j]
            # Force magnitude: power law
            mag = power / (dist ** (power + 1))
            forces[i] += mag * diff
            forces[j] -= mag * diff
        
        # Wall forces
        for axis in range(2):
            # Left wall (x = -0.5)
            dist_left = centers[:, axis] + 0.5
            close_left = dist_left < 0.1
            if np.any(close_left):
                forces[close_left, axis] += power / (np.maximum(dist_left[close_left], 0.01) ** (power + 1))
            
            # Right wall (x = 0.5)
            dist_right = 0.5 - centers[:, axis]
            close_right = dist_right < 0.1
            if np.any(close_right):
                forces[close_right, axis] -= power / (np.maximum(dist_right[close_right], 0.01) ** (power + 1))
        
        # Update positions
        centers += step_size * forces
        
        # Clamp to square
        centers = np.clip(centers, -0.5, 0.5)
    
    return centers


def _local_search(centers, power=4, step_size=0.0005, max_iter=3000):
    """
    More careful local search with adaptive step size.
    """
    centers = np.array(centers, dtype=np.float64)
    best_energy = _repulsion_energy(centers, power)
    best_centers = centers.copy()
    
    lr = step_size
    for it in range(max_iter):
        if time.time() > _DEADLINE:
            break
        
        d = cdist(centers, centers)
        np.fill_diagonal(d, np.inf)
        
        cutoff = 0.2
        mask = (d < cutoff) & (d > 1e-10)
        if not np.any(mask):
            break
        
        forces = np.zeros_like(centers)
        idx_i, idx_j = np.where(mask)
        for i, j in zip(idx_i, idx_j):
            diff = centers[i] - centers[j]
            dist = d[i, j]
            mag = power / (dist ** (power + 1))
            forces[i] += mag * diff
            forces[j] -= mag * diff
        
        for axis in range(2):
            dist_left = centers[:, axis] + 0.5
            close_left = dist_left < 0.1
            if np.any(close_left):
                forces[close_left, axis] += power / (np.maximum(dist_left[close_left], 0.01) ** (power + 1))
            
            dist_right = 0.5 - centers[:, axis]
            close_right = dist_right < 0.1
            if np.any(close_right):
                forces[close_right, axis] -= power / (np.maximum(dist_right[close_right], 0.01) ** (power + 1))
        
        # Normalize forces for stable steps
        force_norm = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norm = np.maximum(force_norm, 1e-10)
        normalized_forces = forces / force_norm
        
        centers_new = centers + lr * normalized_forces
        centers_new = np.clip(centers_new, -0.5, 0.5)
        
        new_energy = _repulsion_energy(centers_new, power)
        if new_energy < best_energy:
            best_energy = new_energy
            best_centers = centers_new.copy()
            centers = centers_new
            # Slightly increase learning rate
            lr = min(lr * 1.01, 0.002)
        else:
            centers = centers_new  # still move, but don't update best
            lr = max(lr * 0.95, 0.0001)
    
    return best_centers


def construct():
    """
    Construct centers of 200 equal circles in [-0.5, 0.5]^2 maximizing the common radius.
    Uses repulsion-based optimization with multiple restarts.
    """
    global _DEADLINE
    _DEADLINE = time.time() + 140.0
    
    best_centers = None
    best_radius = 0.0
    
    # Start with grid
    grid = _initial_grid(N)
    r = _compute_radius(grid)
    if r > best_radius:
        best_radius = r
        best_centers = grid.copy()
    
    # Try several initializations and optimize
    random.seed(42)
    np.random.seed(42)
    
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > _DEADLINE - 10:
            break
        
        if restart == 0:
            centers = grid.copy()
        elif restart == 1:
            centers = _initial_random(N, seed=restart * 1000 + 1)
        else:
            # Perturbed grid
            centers = grid.copy()
            rng = np.random.default_rng(restart * 1000 + 1)
            centers += rng.normal(0, 0.02, size=(N, 2))
            centers = np.clip(centers, -0.5, 0.5)
        
        # Optimize with local search
        centers_opt = _local_search(centers, power=4, step_size=0.0005, max_iter=2000)
        
        # Additional refinement with smaller steps
        centers_opt = _local_search(centers_opt, power=4, step_size=0.0002, max_iter=1500)
        
        r = _compute_radius(centers_opt)
        if r > best_radius:
            best_radius = r
            best_centers = centers_opt.copy()
        
        # Also try a second pass on the best so far if time permits
        if time.time() > _DEADLINE - 20:
            break
    
    # Final refinement on the best result
    if best_centers is not None and time.time() < _DEADLINE - 5:
        final = _local_search(best_centers, power=4, step_size=0.0001, max_iter=1000)
        r_final = _compute_radius(final)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = final
    
    if best_centers is None:
        return grid[:N]
    
    return np.asarray(best_centers, dtype=np.float64)
# EVOLVE-BLOCK-END
