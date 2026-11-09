"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(c):
    """Compute the radius for a set of centers."""
    # Min distance between any pair of centers / 2
    d = np.linalg.norm(c[:, np.newaxis, :] - c[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = np.min(d)
    # Min distance to walls
    min_wall_dist = np.min(0.5 - np.abs(c))
    return min(min_pair_dist / 2.0, min_wall_dist)


def _score_fast(c):
    """Faster score using only the minimum, avoiding full pairwise if possible."""
    # Compute all pairwise distances
    d = np.sqrt(np.sum((c[:, np.newaxis, :] - c[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair_dist = np.min(d)
    min_wall_dist = np.min(0.5 - np.abs(c))
    return min(min_pair_dist / 2.0, min_wall_dist)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin
    
    # Start with a reasonable initial configuration
    # Use a body-centered cubic-like arrangement or random start
    np.random.seed(42)
    
    # Initial: place spheres in a grid, then optimize
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    best_c = grid[:N].astype(np.float64)
    best_score = _score(best_c)
    
    # Simulated annealing
    T = 0.01  # initial temperature
    T_min = 1e-8
    cooling = 0.995
    step_size = 0.05
    
    c = best_c.copy()
    current_score = best_score
    
    for it in range(500000):
        if time.time() > deadline:
            break
        
        # Generate a random perturbation
        idx = np.random.randint(N)
        delta = np.random.normal(0, step_size, size=3)
        new_c = c.copy()
        new_c[idx] += delta
        
        # Check bounds - if out of bounds, reflect or clamp
        if np.any(np.abs(new_c[idx]) > 0.5):
            # Reflect back into bounds
            for k in range(3):
                if new_c[idx, k] > 0.5:
                    new_c[idx, k] = 0.5 - (new_c[idx, k] - 0.5) * 0.5
                elif new_c[idx, k] < -0.5:
                    new_c[idx, k] = -0.5 + (-0.5 - new_c[idx, k]) * 0.5
            new_c[idx] = np.clip(new_c[idx], -0.5, 0.5)
        
        new_score = _score(new_c)
        
        # Accept if better, or with probability exp((new-old)/T)
        if new_score > current_score:
            c = new_c
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_c = c.copy()
        else:
            delta_e = new_score - current_score
            if np.random.random() < np.exp(delta_e / T):
                c = new_c
                current_score = new_score
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Occasionally reheat to escape local minima
        if it % 5000 == 0 and it > 0:
            T = max(T, 0.005)
            step_size = 0.03
    
    # Final local optimization: try to push each sphere outward
    for pass_num in range(50):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            # Try moving sphere i in each direction
            for axis in range(3):
                for direction in [-1, 1]:
                    for eps in [0.001, 0.005, 0.01, 0.02]:
                        new_c = best_c.copy()
                        new_c[i, axis] += direction * eps
                        if np.any(np.abs(new_c[i]) > 0.5):
                            continue
                        new_score = _score(new_c)
                        if new_score > best_score:
                            best_c = new_c
                            best_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        if not improved:
            break
    
    # Ensure final validity
    best_c = np.clip(best_c, -0.4999, 0.4999)
    
    # Verify the result is valid
    final_score = _score(best_c)
    if final_score <= 0 or not np.all(np.isfinite(best_c)):
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_c
# EVOLVE-BLOCK-END
