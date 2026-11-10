"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _radius_of_placement(centers):
    """Compute the largest common radius for a given set of centers."""
    # Distance from each center to the nearest wall
    # Square is [-0.5, 0.5]^2
    dist_to_wall = np.min([
        0.5 - centers[:, 0],  # right wall
        centers[:, 0] + 0.5,  # left wall
        0.5 - centers[:, 1],  # top wall
        centers[:, 1] + 0.5   # bottom wall
    ], axis=0)
    min_wall = np.min(dist_to_wall)
    
    # Minimum pairwise distance / 2
    # For efficiency, we can use a grid-based approach or just compute all pairs
    # For N=200, O(N^2) is 40000 ops which is fine
    if len(centers) <= 1:
        min_pair = np.inf
    else:
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist) / 2.0
    
    return min(min_wall, min_pair)


def _construct_grid():
    """Simple grid construction as baseline."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _local_optimization(centers, max_iters=500, step_scale=0.01):
    """
    Simple gradient-free local optimization using random perturbations.
    Uses the radius as objective (maximize).
    """
    current = centers.copy()
    current_r = _radius_of_placement(current)
    best = current.copy()
    best_r = current_r
    
    step_size = step_scale
    no_improve_count = 0
    
    for it in range(max_iters):
        # Try a random perturbation
        perturbation = np.random.randn(len(current), 2) * step_size
        candidate = current + perturbation
        
        # Ensure points stay in the square (clip slightly inside)
        candidate = np.clip(candidate, -0.5, 0.5)
        
        candidate_r = _radius_of_placement(candidate)
        
        if candidate_r > current_r:
            current = candidate
            current_r = candidate_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Reduce step size if no improvement
            if no_improve_count % 20 == 0:
                step_size *= 0.9
        
        # Keep step size reasonable
        step_size = max(step_size, 1e-6)
    
    return best


def _swap_local_search(centers, max_swaps=200):
    """
    Try swapping positions of pairs of circles to improve packing.
    """
    current = centers.copy()
    current_r = _radius_of_placement(current)
    best = current.copy()
    best_r = current_r
    
    for _ in range(max_swaps):
        # Pick two random indices
        i, j = np.random.choice(len(current), 2, replace=False)
        
        # Try swapping their positions
        candidate = current.copy()
        candidate[i], candidate[j] = current[j], current[i]
        
        candidate_r = _radius_of_placement(candidate)
        
        if candidate_r > current_r:
            current = candidate
            current_r = candidate_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
    
    return best


def _hill_climbing(centers, max_iters=3000):
    """
    More aggressive local optimization using coordinate-wise moves.
    """
    current = centers.copy()
    current_r = _radius_of_placement(current)
    best = current.copy()
    best_r = current_r
    
    step_size = 0.005
    no_improve = 0
    
    for it in range(max_iters):
        # Pick a random point and coordinate
        idx = np.random.randint(len(current))
        coord = np.random.randint(2)
        
        # Try moving in positive and negative directions
        improved = False
        for direction in [1, -1]:
            candidate = current.copy()
            candidate[idx, coord] += direction * step_size
            candidate[idx, coord] = np.clip(candidate[idx, coord], -0.5, 0.5)
            
            candidate_r = _radius_of_placement(candidate)
            if candidate_r > current_r:
                current = candidate
                current_r = candidate_r
                improved = True
                break
        
        if improved:
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve % 100 == 0:
                step_size *= 0.85
                if step_size < 1e-7:
                    break
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Start with a grid
    centers = _construct_grid()
    best_centers = centers.copy()
    best_r = _radius_of_placement(centers)
    
    # Phase 1: Local optimization on the grid
    if time.time() < deadline - 30:
        centers = _local_optimization(centers, max_iters=300, step_scale=0.01)
        r = _radius_of_placement(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # Phase 2: Hill climbing
    if time.time() < deadline - 20:
        centers = _hill_climbing(centers, max_iters=2000)
        r = _radius_of_placement(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # Phase 3: Multiple random restarts with local optimization
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        n_restarts = min(int(remaining_time / 5), 20)  # ~5s per restart
        for restart in range(n_restarts):
            if time.time() >= deadline - 5:
                break
            
            # Random initialization within the square
            rand_centers = np.random.uniform(-0.45, 0.45, (N, 2))
            
            # Quick local optimization
            rand_centers = _local_optimization(rand_centers, max_iters=100, step_scale=0.02)
            r = _radius_of_placement(rand_centers)
            
            if r > best_r:
                best_r = r
                best_centers = rand_centers.copy()
    
    # Final check
    final_r = _radius_of_placement(best_centers)
    if final_r > best_r:
        best_r = final_r
    
    return best_centers
# EVOLVE-BLOCK-END
