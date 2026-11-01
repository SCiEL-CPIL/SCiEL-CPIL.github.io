"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100
CUBE_HALF = 0.5


def _score(c):
    """Compute the radius for a given configuration."""
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check boundaries
    if np.any(c < -CUBE_HALF - 1e-12) or np.any(c > CUBE_HALF + 1e-12):
        return 0.0
    # Distance to walls
    wall_dists = np.minimum(CUBE_HALF - np.abs(c), axis=1)
    r_wall = np.min(wall_dists)
    # Pairwise distances
    # Use vectorized computation for speed
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    r_pair = min_pair / 2.0
    return min(r_wall, r_pair)


def _initial_random(seed=42):
    rng = np.random.RandomState(seed)
    c = rng.uniform(-CUBE_HALF, CUBE_HALF, size=(N, 3))
    return c


def _initial_grid():
    # 5x5x4 grid arrangement (100 points)
    # Side length 1, so spacing in 5x5x4:
    # x: 5 positions, y: 5 positions, z: 4 positions
    xs = np.linspace(-0.5 + 0.5/5, 0.5 - 0.5/5, 5)
    ys = np.linspace(-0.5 + 0.5/5, 0.5 - 0.5/5, 5)
    zs = np.linspace(-0.5 + 0.5/4, 0.5 - 0.5/4, 4)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid[:N]


def _local_search(c, max_iter=2000, step_size=0.01, learning_rate=0.001, seed=0):
    """Simulated annealing / gradient-free local search."""
    rng = np.random.RandomState(seed)
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    temp = 0.005
    cooling = 0.999
    
    for it in range(max_iter):
        # Generate perturbation
        delta = rng.normal(0, step_size, size=(N, 3))
        new_c = c + delta
        
        # Keep within bounds
        new_c = np.clip(new_c, -CUBE_HALF, CUBE_HALF)
        
        new_score = _score(new_c)
        
        if new_score > current_score:
            c = new_c
            current_score = new_score
            if current_score > best_score:
                best_c = c.copy()
                best_score = current_score
        else:
            # Accept with probability based on temperature
            delta_score = new_score - current_score
            if temp > 1e-8:
                prob = np.exp(delta_score / temp)
                if rng.random() < prob:
                    c = new_c
                    current_score = new_score
        
        temp *= cooling
        step_size *= 0.9999
        
        if temp < 1e-10:
            break
    
    return best_c, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Start with grid initialization
    c = _initial_grid()
    best_c = c.copy()
    best_score = _score(best_c)
    
    # Try multiple local searches from different starting points
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        # Start from grid with small random perturbation
        rng = np.random.RandomState(seed)
        c_init = best_c + rng.normal(0, 0.005, size=(N, 3))
        c_init = np.clip(c_init, -CUBE_HALF, CUBE_HALF)
        
        c_found, score_found = _local_search(
            c_init, 
            max_iter=3000, 
            step_size=0.008, 
            seed=seed
        )
        
        if score_found > best_score:
            best_c = c_found
            best_score = score_found
    
    # Final refinement on best solution
    if time.time() < deadline - 10:
        c_found, score_found = _local_search(
            best_c,
            max_iter=5000,
            step_size=0.003,
            seed=9999
        )
        if score_found > best_score:
            best_c = c_found
            best_score = score_found
    
    return best_c
