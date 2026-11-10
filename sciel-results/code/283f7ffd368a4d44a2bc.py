"""Improved packing of 50 equal spheres in a unit cube using FFD + local search."""
import numpy as np
import time
import math
from itertools import combinations

N = 50
TARGET_RADIUS = 0.1359545126417

# EVOLVE-BLOCK-START
def _pairwise_distances(centers):
    """Compute all pairwise distances between centers."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    return dists

def _min_pair_dist(centers):
    """Return minimum pairwise distance."""
    n = len(centers)
    if n < 2:
        return float('inf')
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    return np.min(dists[iu])

def _min_wall_dist(centers):
    """Return minimum distance from any center to any wall (half of clearance)."""
    # Cube is [-0.5, 0.5]^3
    # Distance to nearest wall for point c is min(0.5-c[0], 0.5+c[0], 0.5-c[1], 0.5+c[1], 0.5-c[2], 0.5+c[2])
    coords = np.abs(centers)
    dists = 0.5 - coords
    return np.min(dists)

def _score(centers):
    """Return the radius: min(min_pair_dist/2, min_wall_dist)."""
    mpd = _min_pair_dist(centers)
    mwd = _min_wall_dist(centers)
    return min(mpd / 2.0, mwd)

def _is_valid(centers):
    """Check if centers are valid: finite, inside cube."""
    if centers.shape != (N, 3):
        return False
    if not np.all(np.isfinite(centers)):
        return False
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return False
    return True

def _ffd_packing():
    """First-Fit Decreasing heuristic: place spheres greedily."""
    # We don't have a fixed order of sizes since all spheres are equal.
    # Instead, we use a random sequential placement with some structure.
    # Start with a simple grid and then perturb.
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    return centers

def _local_search(centers, max_iter=10000, step_size=0.01, seed=42):
    """Steepest ascent local search to maximize the minimum distance."""
    rng = np.random.RandomState(seed)
    current = centers.copy()
    best = current.copy()
    best_score = _score(best)
    
    # Precompute neighbor structure? No, just iterate.
    # Use a simple perturbation approach: pick a random point, move it randomly,
    # accept if score improves.
    
    for it in range(max_iter):
        idx = rng.randint(N)
        # Random perturbation
        delta = rng.uniform(-step_size, step_size, size=3)
        candidate = current.copy()
        candidate[idx] += delta
        
        # Check bounds
        if np.any(np.abs(candidate[idx]) > 0.5):
            continue
        
        new_score = _score(candidate)
        if new_score > best_score:
            best = candidate.copy()
            best_score = new_score
            current = candidate
        else:
            # Random restarts of perturbation direction? Just skip for now.
            pass
            
    return best, best_score

def _optimize_with_simulated_annealing(centers, max_iter=50000, start_temp=0.05, end_temp=0.001, seed=42):
    """Simulated annealing to maximize the minimum distance."""
    rng = np.random.RandomState(seed)
    current = centers.copy()
    best = current.copy()
    best_score = _score(best)
    current_score = best_score
    
    for it in range(max_iter):
        # Annealing schedule
        temp = start_temp * (end_temp / start_temp) ** (it / max_iter)
        
        idx = rng.randint(N)
        delta = rng.uniform(-start_temp, start_temp, size=3) * 0.5  # Scale down
        candidate = current.copy()
        candidate[idx] += delta
        
        if np.any(np.abs(candidate[idx]) > 0.5):
            continue
        
        new_score = _score(candidate)
        delta_score = new_score - current_score
        
        if delta_score > 0 or rng.random() < math.exp(delta_score / temp):
            current = candidate
            current_score = new_score
            if new_score > best_score:
                best = candidate.copy()
                best_score = new_score
                
    return best, best_score

def construct():
    start_time = time.time()
    deadline = 140.0  # 20s safety margin
    
    # Start with FFD grid
    centers = _ffd_packing()
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    # Run multiple rounds of local search with different initial perturbations
    num_rounds = 5
    for r in range(num_rounds):
        if time.time() > deadline:
            break
        
        # Perturb the current best slightly
        rng = np.random.RandomState(r)
        perturbed = best_centers + rng.uniform(-0.01, 0.01, size=(N, 3))
        # Clamp to cube
        perturbed = np.clip(perturbed, -0.5, 0.5)
        
        # Local search
        ls_centers, ls_score = _local_search(perturbed, max_iter=2000, step_size=0.02, seed=r)
        if ls_score > best_score:
            best_centers = ls_centers
            best_score = ls_score
        
        # Simulated annealing on top
        if time.time() > deadline:
            break
        sa_centers, sa_score = _optimize_with_simulated_annealing(ls_centers, max_iter=10000, start_temp=0.03, end_temp=0.0005, seed=r+100)
        if sa_score > best_score:
            best_centers = sa_centers
            best_score = sa_score
    
    return best_centers
# EVOLVE-BLOCK-END
