"""Optimized packing of 50 equal circles in a unit square."""
import numpy as np
from scipy.spatial import cKDTree
import time

N = 50

# EVOLVE-BLOCK-START
def _score(centers):
    """Compute the maximum radius for given centers."""
    # Distance to walls
    wall_dists = np.min(np.stack([
        0.5 - centers[:, 0],
        0.5 + centers[:, 0],
        0.5 - centers[:, 1],
        0.5 + centers[:, 1]
    ], axis=1), axis=1)
    
    # Minimum inter-circle distance
    tree = cKDTree(centers)
    pairs = tree.query_pairs(1.0, output_type='ndarray')
    if len(pairs) == 0:
        min_pair_dist = np.inf
    else:
        diffs = centers[pairs[:, 0]] - centers[pairs[:, 1]]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        min_pair_dist = np.min(dists)
    
    return np.min([np.min(wall_dists), min_pair_dist / 2.0])


def _heuristic_init(seed):
    """Generate an initial configuration using a heuristic."""
    rng = np.random.RandomState(seed)
    # Start with a grid-like arrangement with some randomness
    m = 7
    # Create a 7x7 grid and pick 50 points
    xs = np.linspace(-0.42, 0.42, m)
    ys = np.linspace(-0.42, 0.42, m)
    grid = np.array([(x, y) for y in ys for x in xs])
    # Shuffle and pick first N
    idx = rng.permutation(len(grid))[:N]
    centers = grid[idx].copy()
    # Add small random perturbation
    centers += rng.uniform(-0.01, 0.01, size=centers.shape)
    # Clip to valid region
    centers = np.clip(centers, -0.45, 0.45)
    return centers


def _local_search(centers, max_iter=5000, step_scale=0.05, rng=None):
    """Local search to improve the packing."""
    if rng is None:
        rng = np.random.RandomState()
    
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    for it in range(max_iter):
        # Try to move one circle
        i = rng.randint(N)
        # Random step
        step = rng.normal(0, step_scale, size=2)
        new_centers = best_centers.copy()
        new_centers[i] += step
        
        # Check if still inside square
        if np.any(np.abs(new_centers) > 0.48):
            continue
            
        new_score = _score(new_centers)
        if new_score > best_score:
            best_score = new_score
            best_centers = new_centers.copy()
            step_scale = max(0.005, step_scale * 0.99)
        else:
            step_scale = min(0.1, step_scale * 1.01)
            
        # Occasionally try larger moves
        if it % 100 == 0:
            step_scale = 0.05
            
    return best_centers, best_score


def construct():
    """Main entry point: find optimal packing of 50 circles in unit square."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_centers = None
    best_score = -1.0
    
    # Try multiple restarts with different seeds
    for seed in range(200):
        if time.time() > deadline:
            break
            
        centers = _heuristic_init(seed)
        improved_centers, score = _local_search(
            centers, 
            max_iter=2000, 
            step_scale=0.03,
            rng=np.random.RandomState(seed + 1000)
        )
        
        if score > best_score:
            best_score = score
            best_centers = improved_centers.copy()
            
        # If we're getting close to known best, do more fine-tuning
        if best_score > 0.070:
            # Do a more careful local search
            for sub_seed in range(5):
                if time.time() > deadline:
                    break
                fine_centers, fine_score = _local_search(
                    best_centers + np.random.RandomState(seed + sub_seed).normal(0, 0.01, size=(N, 2)),
                    max_iter=3000,
                    step_scale=0.01,
                    rng=np.random.RandomState(seed + sub_seed + 2000)
                )
                if fine_score > best_score:
                    best_score = fine_score
                    best_centers = fine_centers.copy()
    
    # Final validation
    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    # Ensure all centers are strictly inside the square
    best_centers = np.clip(best_centers, -0.499, 0.499)
    
    return best_centers
# EVOLVE-BLOCK-END
