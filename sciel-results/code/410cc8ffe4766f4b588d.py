"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    n = len(centres)
    # Wall distance
    wall = 0.5 - np.abs(centres)
    min_wall = np.min(wall)
    # Pairwise distances
    # For N=100, this is 100*100/2 = 4950 pairs, manageable
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(min_wall, min_pair / 2.0)


def _score(centres):
    """Score is the radius."""
    return _compute_radius(centres)


def _refine_spectral(centres, n_iter=50, step_size=0.01):
    """
    Refine centres using a spectral/gradient-like approach.
    We maximize the minimum distance by iteratively pushing apart close pairs.
    """
    n = len(centres)
    centres = centres.copy()
    
    for it in range(n_iter):
        # Compute all pairwise distances
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Find the minimum distance pair
        min_dist = np.min(dist)
        if min_dist < 1e-15:
            continue
            
        # Compute forces: push apart pairs that are too close
        # Force on i from j is proportional to (direction) / dist^2 for close pairs
        # But we want to maximize the minimum, so we focus on the closest pairs
        
        # Compute gradient of the minimum distance
        # For each pair, the contribution to the gradient of the min is only if it's the min
        # This is complex, so we use a simpler heuristic:
        # Push all pairs apart with a force that is stronger for closer pairs
        
        forces = np.zeros_like(centres)
        for i in range(n):
            for j in range(i + 1, n):
                d = dist[i, j]
                if d < 0.3:  # Only consider pairs that are relatively close
                    # Force magnitude inversely proportional to distance
                    f_mag = step_size / (d + 1e-10)
                    direction = (centres[i] - centres[j]) / d
                    forces[i] += f_mag * direction
                    forces[j] -= f_mag * direction
        
        # Also push away from walls
        for i in range(n):
            for k in range(3):
                c = centres[i, k]
                if c > 0:  # Right wall
                    forces[i, k] -= step_size / (0.5 - c + 1e-10)
                elif c < 0:  # Left wall
                    forces[i, k] += step_size / (0.5 + c + 1e-10)
        
        # Apply forces with adaptive step
        centres += forces * 0.1
        
        # Clip to stay inside
        centres = np.clip(centres, -0.5 + 1e-6, 0.5 - 1e-6)
    
    return centres


def _local_search(centres, n_moves=200, time_limit=10):
    """
    Simple local search: try moving each centre randomly and keep if improved.
    """
    n = len(centres)
    centres = centres.copy()
    best_score = _score(centres)
    start_time = time.time()
    
    for move in range(n_moves):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random centre to move
        i = np.random.randint(n)
        old_pos = centres[i].copy()
        
        # Random perturbation
        delta = np.random.normal(0, 0.01, 3)
        centres[i] += delta
        centres[i] = np.clip(centres[i], -0.5 + 1e-6, 0.5 - 1e-6)
        
        new_score = _score(centres)
        if new_score > best_score:
            best_score = new_score
        else:
            centres[i] = old_pos
    
    return centres, best_score


def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for the 150s limit
    
    # Initial guess: use a stratified random placement
    # This is better than a grid because it can adapt to local constraints
    
    # Start with a simple grid as a baseline
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centres = grid[:N].copy()
    
    # Add some randomization to break symmetry
    np.random.seed(42)
    centres += np.random.normal(0, 0.01, (N, 3))
    centres = np.clip(centres, -0.5 + 1e-6, 0.5 - 1e-6)
    
    best_centres = centres.copy()
    best_score = _score(centres)
    
    # Phase 1: Spectral refinement
    if time.time() - start_time < deadline - 30:
        centres = _refine_spectral(centres, n_iter=30, step_size=0.005)
        score = _score(centres)
        if score > best_score:
            best_score = score
            best_centres = centres.copy()
    
    # Phase 2: Local search
    if time.time() - start_time < deadline - 5:
        centres, score = _local_search(centres, n_moves=100, time_limit=20)
        if score > best_score:
            best_score = score
            best_centres = centres.copy()
    
    # Phase 3: More refinement if time permits
    if time.time() - start_time < deadline - 2:
        centres = _refine_spectral(best_centres, n_iter=20, step_size=0.002)
        score = _score(centres)
        if score > best_score:
            best_score = score
            best_centres = centres.copy()
    
    # Final check: ensure validity
    # Clip to ensure strictly inside
    best_centres = np.clip(best_centres, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_centres
# EVOLVE-BLOCK-END
