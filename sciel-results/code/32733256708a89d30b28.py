"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_score(centers):
    """
    Compute the packing radius for given centers.
    Returns the minimum of (min_pairwise_dist / 2, min_wall_dist).
    """
    # Wall distances: min over all centers of min over coords of (0.5 - |coord|)
    min_wall = np.min(0.5 - np.abs(centers))
    
    # Pairwise distances
    # For N=50, we can use the vectorized approach
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_sq = np.min(dists_sq)
    min_pair = np.sqrt(min_pair_sq)
    
    return min(min_pair / 2.0, min_wall)


def construct():
    """
    Use simulated annealing to maximize the packing radius.
    Improved version with better initialization, adaptive step size, and multiple restarts.
    """
    rng = np.random.default_rng(42)
    
    start_time = time.time()
    max_time = 110.0  # Leave 10s margin for safety
    
    best_centers = None
    best_score = -np.inf
    
    # Try multiple initialization strategies
    initializations = []
    
    # 1. 7x8 grid (with jitter)
    rows, cols = 7, 8
    x_ticks = np.linspace(-0.45, 0.45, cols)
    y_ticks = np.linspace(-0.45, 0.45, rows)
    centers_grid = np.zeros((N, 2))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < N:
                centers_grid[idx] = [x_ticks[c], y_ticks[r]]
                idx += 1
            else:
                break
        if idx >= N:
            break
    centers_grid += rng.normal(0, 0.01, (N, 2))
    centers_grid = np.clip(centers_grid, -0.4999, 0.4999)
    initializations.append(centers_grid)
    
    # 2. 5x10 grid (with jitter)
    rows, cols = 5, 10
    x_ticks = np.linspace(-0.48, 0.48, cols)
    y_ticks = np.linspace(-0.48, 0.48, rows)
    centers_grid2 = np.zeros((N, 2))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < N:
                centers_grid2[idx] = [x_ticks[c], y_ticks[r]]
                idx += 1
            else:
                break
        if idx >= N:
            break
    centers_grid2 += rng.normal(0, 0.01, (N, 2))
    centers_grid2 = np.clip(centers_grid2, -0.4999, 0.4999)
    initializations.append(centers_grid2)
    
    # 3. Random initialization
    centers_random = rng.uniform(-0.45, 0.45, (N, 2))
    initializations.append(centers_random)
    
    # Run SA from each initialization
    for init_centers in initializations:
        if time.time() - start_time > max_time:
            break
        
        centers = init_centers.copy()
        current_score = _compute_score(centers)
        
        best_centers_local = centers.copy()
        best_score_local = current_score
        
        # SA parameters
        n_iterations = 1000000
        T_start = 0.03
        T_end = 1e-10
        alpha = (T_end / T_start) ** (1.0 / n_iterations)
        
        base_step = 0.04
        step_size = base_step
        
        no_improve_count = 0
        max_no_improve = 5000
        
        for it in range(n_iterations):
            # Check time
            if it % 5000 == 0:
                if time.time() - start_time > max_time:
                    break
            
            T = T_start * (alpha ** it)
            
            # Pick a random center
            i = rng.integers(N)
            old_pos = centers[i].copy()
            
            # Perturb with adaptive step size
            centers[i] += rng.normal(0, step_size, 2)
            centers[i] = np.clip(centers[i], -0.4999, 0.4999)
            
            # Compute new score
            new_score = _compute_score(centers)
            
            if new_score >= current_score:
                current_score = new_score
                no_improve_count = 0
                if new_score > best_score_local:
                    best_score_local = new_score
                    best_centers_local = centers.copy()
            else:
                delta = new_score - current_score
                # Accept with prob exp(delta/T)
                if delta / T > -100:
                    if rng.random() < np.exp(delta / T):
                        current_score = new_score
                        no_improve_count = 0
                    else:
                        centers[i] = old_pos
                        no_improve_count += 1
                else:
                    centers[i] = old_pos
                    no_improve_count += 1
            
            # Adaptive step size
            if no_improve_count > max_no_improve:
                step_size *= 0.5
                no_improve_count = 0
            elif it % 2000 == 0 and it > 0:
                step_size *= 0.99
                
            # Periodically restart step size to escape local minima
            if it % 20000 == 0 and it > 0:
                step_size = base_step * (0.5 + 0.5 * rng.random())
                no_improve_count = 0
        
        # Update global best
        if best_score_local > best_score:
            best_score = best_score_local
            best_centers = best_centers_local
    
    return best_centers
# EVOLVE-BLOCK-END
