"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
import sys


def _compute_score(centres):
    """Compute the maximum common radius for the given centres."""
    if centres.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check centres are inside the cube [-0.5, 0.5]^3
    if np.any(np.abs(centres) > 0.5 + 1e-12):
        return 0.0
    # Compute pairwise distances
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists)
    # Distance to walls
    wall_dists = 0.5 - np.abs(centres)
    min_wall = np.min(wall_dists)
    return min(min_pairwise / 2.0, min_wall)


def _random_init(n_points, seed=42):
    """Generate random initial positions inside the cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.5, 0.5, size=(n_points, 3))


def _grid_init(n_points):
    """Generate a grid-like initial distribution."""
    # Try to create a roughly uniform grid
    m = int(np.ceil(n_points ** (1.0 / 3.0)))
    # Create all grid points
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # If we have more than needed, randomly select
    if len(grid) >= n_points:
        # Use a stratified random selection for better coverage
        indices = np.random.choice(len(grid), size=n_points, replace=False)
        return grid[indices]
    else:
        # Pad with random points
        remaining = n_points - len(grid)
        extra = np.random.uniform(-0.5, 0.5, size=(remaining, 3))
        return np.vstack([grid, extra])


def _hill_climbing(centres, max_iters=5000, step_size=0.05, lr_decay=0.999, 
                   min_step=0.001, time_limit=100, start_time=None):
    """Perform hill climbing to improve the packing."""
    if start_time is None:
        start_time = time.time()
    
    current = centres.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    rng = np.random.RandomState(12345)
    
    for it in range(max_iters):
        # Check time limit
        if time.time() - start_time > time_limit:
            break
        
        # Generate a random perturbation
        idx = rng.randint(N)
        delta = rng.normal(0, step_size, size=3)
        
        # Try the move
        new_centre = current.copy()
        new_centre[idx] = current[idx] + delta
        
        # Keep inside the cube
        new_centre[idx] = np.clip(new_centre[idx], -0.5, 0.5)
        
        new_score = _compute_score(new_centre)
        
        if new_score > current_score:
            current = new_centre
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        else:
            # Accept with some probability (simulated annealing style)
            if rng.random() < np.exp((new_score - current_score) / (step_size * 0.1)):
                current = new_centre
                current_score = new_score
                if current_score > best_score:
                    best = current.copy()
                    best_score = current_score
        
        # Decay step size
        step_size *= lr_decay
        if step_size < min_step:
            step_size = min_step
    
    return best, best_score


def _local_search(centres, max_iters=3000, time_limit=80, start_time=None):
    """More aggressive local search with multiple restarts."""
    if start_time is None:
        start_time = time.time()
    
    current = centres.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    rng = np.random.RandomState(99999)
    step_size = 0.1
    
    for it in range(max_iters):
        # Check time limit
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random point and try to move it
        idx = rng.randint(N)
        
        # Generate several candidate moves
        for _ in range(5):
            delta = rng.normal(0, step_size, size=3)
            new_centre = current.copy()
            new_centre[idx] = current[idx] + delta
            new_centre[idx] = np.clip(new_centre[idx], -0.5, 0.5)
            
            new_score = _compute_score(new_centre)
            if new_score > current_score:
                current = new_centre
                current_score = new_score
                if current_score > best_score:
                    best = current.copy()
                    best_score = current_score
                break
        
        # Decay step size
        step_size *= 0.9995
        if step_size < 0.005:
            step_size = 0.005
    
    return best, best_score


def construct():
    """Main entry point: find the best packing of 100 equal spheres in a unit cube."""
    start_time = time.time()
    total_time_limit = 140  # Leave 10s margin
    
    # Start with a good initial configuration
    # Use a combination of grid and random initialization
    best_centres = None
    best_score = 0.0
    
    # Try multiple initializations
    initializations = [
        _grid_init(N),
        _random_init(N, seed=1),
        _random_init(N, seed=2),
        _random_init(N, seed=3),
    ]
    
    for init_idx, init_centres in enumerate(initializations):
        if time.time() - start_time > total_time_limit - 10:
            break
        
        # Run local search
        result, score = _local_search(init_centres, 
                                       max_iters=2000, 
                                       time_limit=total_time_limit - 15 - (time.time() - start_time),
                                       start_time=start_time)
        
        if score > best_score:
            best_score = score
            best_centres = result.copy()
    
    # If we have time, do more refinement
    remaining_time = total_time_limit - (time.time() - start_time)
    if remaining_time > 20 and best_centres is not None:
        # Do additional hill climbing
        refined, refined_score = _hill_climbing(
            best_centres, 
            max_iters=3000,
            step_size=0.02,
            time_limit=remaining_time - 5,
            start_time=start_time
        )
        if refined_score > best_score:
            best_score = refined_score
            best_centres = refined
    
    # Final check
    if best_centres is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centres = grid[:N]
        best_score = _compute_score(best_centres)
    
    return best_centres
# EVOLVE-BLOCK-END
