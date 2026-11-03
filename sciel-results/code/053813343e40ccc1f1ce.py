"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.spatial import cKDTree

N = 100

# EVOLVE-BLOCK-START
def _score(centers):
    """Compute the radius for a given set of centers."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    tree = cKDTree(centers)
    pairs = tree.query_pairs(1.0)  # initial threshold, we'll refine
    if not pairs:
        return min_wall
    
    # Get all pairs within a reasonable distance to compute min
    # For efficiency, we can just compute the minimum pairwise distance
    # Using cKDTree for nearest neighbor
    # Actually, let's just compute all pairwise distances for N=100 (4950 pairs)
    dists = []
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            dists.append(d)
    min_pair = min(dists)
    
    r = min(min_pair / 2.0, min_wall)
    return r


def _score_fast(centers):
    """Faster scoring using scipy distance matrix."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    r = min(min_pair / 2.0, min_wall)
    return r


def _initial_grid():
    """Create an initial grid packing."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _simulated_annealing(centers, max_time=100.0):
    """Run simulated annealing to maximize the minimum radius."""
    best_centers = centers.copy()
    best_score = _score_fast(best_centers)
    
    current_centers = centers.copy()
    current_score = best_score
    
    start_time = time.time()
    deadline = start_time + max_time
    
    T = 0.005
    T_min = 1e-8
    cooling = 0.999
    
    np.random.seed(42)
    random.seed(42)
    
    step_size = 0.01
    
    while time.time() < deadline:
        # Check for time
        if time.time() > deadline - 5.0:
            break
        
        # Propose a move: randomly pick a few spheres and perturb them
        n_move = np.random.randint(1, 4)
        indices = np.random.choice(N, n_move, replace=False)
        
        old_positions = current_centers[indices].copy()
        
        # Perturb
        perturb = np.random.randn(n_move, 3) * step_size
        new_positions = old_positions + perturb
        
        # Keep within bounds
        new_positions = np.clip(new_positions, -0.49, 0.49)
        
        current_centers[indices] = new_positions
        
        new_score = _score_fast(current_centers)
        
        # Accept or reject
        if new_score > current_score or np.random.random() < np.exp((new_score - current_score) / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_centers = current_centers.copy()
        else:
            # Reject
            current_centers[indices] = old_positions
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Adapt step size
        if new_score > current_score:
            step_size *= 1.01
        else:
            step_size *= 0.99
        step_size = np.clip(step_size, 0.001, 0.05)
    
    return best_centers, best_score


def _local_optimization(centers):
    """Do a local optimization pass to improve the packing."""
    best_centers = centers.copy()
    best_score = _score_fast(best_centers)
    
    for iteration in range(1000):
        improved = False
        
        # For each pair of closest spheres, try to move them apart
        tree = cKDTree(best_centers)
        pairs = tree.query_pairs(0.25)
        
        if not pairs:
            break
        
        # Find the minimum distance pair
        min_dist = np.inf
        min_pair = None
        for i, j in pairs:
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            if d < min_dist:
                min_dist = d
                min_pair = (i, j)
        
        if min_pair is None:
            break
        
        i, j = min_pair
        direction = best_centers[j] - best_centers[i]
        if np.linalg.norm(direction) < 1e-12:
            direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        
        # Try moving i and j apart
        for delta in [0.001, 0.002, 0.005, 0.01]:
            trial = best_centers.copy()
            trial[i] -= direction * delta
            trial[j] += direction * delta
            trial = np.clip(trial, -0.49, 0.49)
            
            trial_score = _score_fast(trial)
            if trial_score > best_score:
                best_centers = trial
                best_score = trial_score
                improved = True
                break
        
        if not improved:
            # Try random small moves
            for _ in range(10):
                idx = np.random.randint(N)
                trial = best_centers.copy()
                trial[idx] += np.random.randn(3) * 0.005
                trial = np.clip(trial, -0.49, 0.49)
                trial_score = _score_fast(trial)
                if trial_score > best_score:
                    best_centers = trial
                    best_score = trial_score
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_centers, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with grid
    centers = _initial_grid()
    best_centers = centers.copy()
    best_score = _score_fast(best_centers)
    
    # Run simulated annealing with multiple restarts
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            current = centers.copy()
        else:
            # Add noise to previous best
            current = best_centers + np.random.randn(N, 3) * 0.05
            current = np.clip(current, -0.49, 0.49)
        
        # Run SA
        sa_centers, sa_score = _simulated_annealing(current, max_time=20.0)
        
        if sa_score > best_score:
            best_score = sa_score
            best_centers = sa_centers
        
        # Local optimization
        if time.time() > deadline - 10.0:
            break
        
        local_centers, local_score = _local_optimization(best_centers)
        
        if local_score > best_score:
            best_score = local_score
            best_centers = local_centers
    
    # Final check
    if time.time() < deadline - 5.0:
        # One more local optimization pass
        final_centers, final_score = _local_optimization(best_centers)
        if final_score > best_score:
            best_centers = final_centers
            best_score = final_score
    
    return best_centers
# EVOLVE-BLOCK-END
