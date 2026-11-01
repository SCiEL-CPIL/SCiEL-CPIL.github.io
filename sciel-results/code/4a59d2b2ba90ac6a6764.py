"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np

N = 50

def _compute_score(centers):
    """
    Compute the maximum radius r for the given centers.
    r = min(min_pair_dist / 2, min_wall_dist)
    """
    # Pairwise distances
    diff = centers[:, np.newaxis] - centers[np.newaxis, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Ignore self-distances
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    # Distance to nearest wall for point (x,y) in [-0.5, 0.5]^2 is 0.5 - max(|x|, |y|)
    abs_centers = np.abs(centers)
    max_coord = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_coord
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _simulated_annealing(centers_init, n_iterations, T_start, T_end, base_step, rng):
    centers = centers_init.copy()
    current_score = _compute_score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    # Adaptive step size parameters
    step_size = base_step
    acceptance_count = 0
    window_size = 100
    no_improve = 0
    max_no_improve = 2000
    
    for it in range(n_iterations):
        # Cooling
        T = T_start * (T_end / T_start) ** (it / n_iterations)
        
        # Move
        i = rng.integers(N)
        old_pos = centers[i].copy()
        
        # Perturbation: Random Gaussian
        centers[i] += rng.normal(0, step_size, 2)
        
        # Clip to valid range
        centers[i] = np.clip(centers[i], -0.499, 0.499)
        
        new_score = _compute_score(centers)
        
        if new_score >= current_score:
            current_score = new_score
            acceptance_count += 1
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Accept with probability exp((new_score - current_score) / T)
            delta = new_score - current_score
            if delta / T > -100:
                if rng.random() < np.exp(delta / T):
                    current_score = new_score
                    acceptance_count += 1
                    no_improve += 1
                else:
                    centers[i] = old_pos
                    no_improve += 1
            else:
                centers[i] = old_pos
                no_improve += 1
        
        # Adaptive Step Size
        if it % window_size == 0 and it > 0:
            accept_rate = acceptance_count / window_size
            acceptance_count = 0
            
            # Adjust step size to maintain ~50% acceptance
            if accept_rate > 0.7:
                step_size *= 1.1
            elif accept_rate < 0.3:
                step_size *= 0.9
            
            # If stuck, try larger jumps
            if no_improve > max_no_improve:
                step_size *= 1.5
                no_improve = 0
        
    return best_centers, best_score

def _hexagonal_init(rng, jitter=0.0):
    """
    Generate a hexagonal-like initial configuration.
    """
    centers = np.zeros((N, 2))
    
    d = 0.14
    row_spacing = d * np.sqrt(3) / 2
    
    idx = 0
    y = -0.45
    row = 0
    while idx < N and y <= 0.45:
        x_offset = 0.0 if row % 2 == 0 else d / 2
        x = -0.45 + x_offset
        while x <= 0.45 and idx < N:
            centers[idx] = [x, y]
            idx += 1
            x += d
        y += row_spacing
        row += 1
    
    if idx < N:
        remaining = N - idx
        centers[idx:idx+remaining] = rng.uniform(-0.2, 0.2, (remaining, 2))
    
    if jitter > 0:
        centers += rng.normal(0, jitter, (N, 2))
    
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def construct():
    """
    Use simulated annealing with multiple restarts to maximize the packing radius.
    """
    rng = np.random.default_rng(42)
    
    best_overall = None
    best_overall_score = -np.inf
    
    # Parameters
    n_restarts = 6
    n_iters_per_restart = 150000
    T_start = 0.02
    T_end = 1e-10
    base_step = 0.03
    
    for restart in range(n_restarts):
        # Vary initialization slightly
        jitter = 0.005 + 0.005 * restart
        centers_init = _hexagonal_init(rng, jitter=jitter)
        
        # Vary temperature and step slightly
        T_s = T_start * (1.0 + 0.1 * restart)
        step_s = base_step * (1.0 + 0.1 * restart)
        
        best_centers, best_score = _simulated_annealing(
            centers_init, n_iters_per_restart, T_s, T_end, step_s, rng
        )
        
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall = best_centers
            
    return best_overall
# EVOLVE-BLOCK-END
