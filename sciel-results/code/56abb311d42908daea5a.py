"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_score(centers):
    """Compute the minimum radius for the given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Vectorized pairwise distance computation
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)


def _optimize_sa(centers, time_budget, initial_temp=0.005, final_temp=1e-6, cooling=0.9995):
    """Simulated annealing to maximize minimum radius."""
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_score = _compute_score(current)
    current_score = best_score
    
    temp = initial_temp
    step_scale = 0.02
    
    while temp > final_temp:
        if time.time() - start_time > time_budget:
            break
        
        # Pick a random sphere and perturb
        idx = np.random.randint(N)
        delta = np.random.randn(3) * step_scale * temp / initial_temp
        new_centers = current.copy()
        new_centers[idx] += delta
        
        # Clip to bounds
        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_score = _compute_score(new_centers)
        delta_score = new_score - current_score
        
        # Accept if better or with probability exp(delta_score/temp)
        if delta_score > 0 or np.random.random() < np.exp(delta_score / temp):
            current = new_centers
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = current.copy()
        
        temp *= cooling
    
    return best, best_score


def _bcc_initial():
    """Generate a BCC-like initial configuration for 50 spheres."""
    # Use a 4x4x4 grid = 64 positions, take 50
    m = 4
    spacing = 1.0 / m
    centers = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = -0.5 + (i + 0.5) * spacing
                y = -0.5 + (j + 0.5) * spacing
                z = -0.5 + (k + 0.5) * spacing
                centers.append([x, y, z])
    
    centers = np.array(centers[:N])
    return centers


def construct():
    """Construct optimal sphere packing."""
    start_time = time.time()
    deadline = 120.0  # safety margin before 150s kill
    
    np.random.seed(42)
    
    # Generate initial configuration
    initial = _bcc_initial()
    
    best_centers = initial.copy()
    best_score = _compute_score(initial)
    
    # Multi-restart simulated annealing
    num_restarts = 6
    
    for restart in range(num_restarts):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining < 10.0:
            break
        
        # Allocate time for this restart
        time_per_restart = min(remaining / (num_restarts - restart), 15.0)
        
        if restart == 0:
            centers = initial.copy()
        else:
            # Perturb the initial configuration
            perturbation_scale = 0.01 + 0.005 * restart
            centers = initial + np.random.normal(0, perturbation_scale, size=initial.shape)
            centers = np.clip(centers, -0.49, 0.49)
        
        optimized, score = _optimize_sa(centers, time_budget=time_per_restart)
        
        if score > best_score:
            best_score = score
            best_centers = optimized.copy()
    
    # Final polish: short SA run on the best result
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 5.0:
        polished, polished_score = _optimize_sa(best_centers, time_budget=min(remaining - 2.0, 10.0), initial_temp=0.002)
        if polished_score > best_score:
            best_score = polished_score
            best_centers = polished.copy()
    
    # Ensure valid output
    best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_centers
# EVOLVE-BLOCK-END
