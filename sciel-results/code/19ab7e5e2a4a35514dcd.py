"""Improved packing of 50 equal spheres in a unit cube using KDTree-accelerated search."""
import numpy as np
import time
import math
from scipy.spatial import cKDTree

N = 50

# EVOLVE-BLOCK-START
def _min_pair_dist(centers):
    """Return minimum pairwise distance using KDTree."""
    n = len(centers)
    if n < 2:
        return float('inf')
    tree = cKDTree(centers)
    # Query k=2 to get self and nearest neighbor
    dists, idxs = tree.query(centers, k=2)
    # dists[:, 1] is the distance to the nearest neighbor
    return np.min(dists[:, 1])

def _min_wall_dist(centers):
    """Return minimum distance from any center to any wall."""
    # Cube is [-0.5, 0.5]^3
    # Distance to nearest wall for point c is 0.5 - max(|c_x|, |c_y|, |c_z|)
    coords = np.abs(centers)
    max_coords = np.max(coords, axis=1)
    dists = 0.5 - max_coords
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
    """First-Fit Decreasing heuristic: place spheres greedily on a grid."""
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    return centers

def _local_search(centers, time_limit, seed=42):
    """
    Constraint-aware local search.
    Identifies the bottleneck (close pair or wall violation) and moves points to fix it.
    """
    rng = np.random.RandomState(seed)
    current = centers.copy()
    best = current.copy()
    best_score = _score(best)
    
    start_time = time.time()
    step_size = 0.01
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit:
        mpd = _min_pair_dist(current)
        mwd = _min_wall_dist(current)
        current_score = min(mpd / 2.0, mwd)
        
        if current_score > best_score:
            best_score = current_score
            best = current.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturb to escape local minimum
                current += rng.normal(0, 0.005, size=(N, 3))
                current = np.clip(current, -0.5, 0.5)
                no_improve_count = 0
                # Reduce step size if we've been stuck for a long time? 
                # Or increase to jump out? Let's try increasing slightly then decaying.
                step_size = 0.02
                continue
        
        # Determine bottleneck
        if mpd / 2.0 < mwd:
            # Pair distance is bottleneck
            tree = cKDTree(current)
            dists, idxs = tree.query(current, k=2)
            nn_dists = dists[:, 1]
            
            # Find points with small nearest neighbor distances
            # We want to move points that are part of the minimum distance pair
            min_nn = np.min(nn_dists)
            threshold = min_nn + 0.001
            candidates = np.where(nn_dists < threshold)[0]
            
            if len(candidates) > 0:
                idx = candidates[rng.randint(len(candidates))]
                nn_idx = idxs[idx, 1]
                dist = nn_dists[idx]
                
                if dist > 1e-9:
                    direction = (current[idx] - current[nn_idx]) / dist
                    current[idx] += direction * step_size
                else:
                    current[idx] += rng.normal(0, step_size, 3)
            else:
                # Fallback: move random point away from its nearest neighbor
                idx = rng.randint(N)
                nn_idx = idxs[idx, 1]
                dist = nn_dists[idx]
                if dist > 1e-9:
                    direction = (current[idx] - current[nn_idx]) / dist
                    current[idx] += direction * step_size
                else:
                    current[idx] += rng.normal(0, step_size, 3)
        else:
            # Wall distance is bottleneck
            wall_dists = 0.5 - np.max(np.abs(current), axis=1)
            min_wall_val = np.min(wall_dists)
            threshold = min_wall_val + 0.001
            candidates = np.where(wall_dists < threshold)[0]
            
            if len(candidates) > 0:
                idx = candidates[rng.randint(len(candidates))]
            else:
                idx = rng.randint(N)
            
            # Move towards center
            norm = np.linalg.norm(current[idx])
            if norm > 1e-9:
                direction = -current[idx] / norm
            else:
                direction = rng.normal(0, 1, 3)
                direction /= np.linalg.norm(direction)
            
            current[idx] += direction * step_size
            
        # Clamp to cube
        current = np.clip(current, -0.5, 0.5)
        
        # Decay step size slowly
        step_size *= 0.999
        if step_size < 0.001:
            step_size = 0.01 # Reset to prevent stalling

    return best, best_score

def _optimize_with_sa(centers, time_limit, seed=42):
    """Simulated Annealing with fast scoring."""
    rng = np.random.RandomState(seed)
    current = centers.copy()
    best = current.copy()
    best_score = _score(best)
    current_score = best_score
    
    start_time = time.time()
    start_temp = 0.05
    end_temp = 0.001
    elapsed = 0
    
    # Estimate iterations based on time limit? 
    # Just run until time limit.
    while time.time() - start_time < time_limit:
        # Linear cooling
        progress = min(1.0, (time.time() - start_time) / time_limit)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        idx = rng.randint(N)
        delta = rng.uniform(-0.02, 0.02, size=3)
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
    deadline = 110.0  # Safety margin for 120s limit
    
    # Start with FFD grid
    centers = _ffd_packing()
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    # Run multiple rounds of local search with different initial perturbations
    num_rounds = 10
    for r in range(num_rounds):
        if time.time() > deadline:
            break
        
        # Perturb the current best slightly
        rng = np.random.RandomState(r)
        perturbed = best_centers + rng.uniform(-0.02, 0.02, size=(N, 3))
        perturbed = np.clip(perturbed, -0.5, 0.5)
        
        # Local search
        ls_centers, ls_score = _local_search(perturbed, time_limit=10.0, seed=r)
        if ls_score > best_score:
            best_centers = ls_centers
            best_score = ls_score
        
        # Simulated annealing on top
        if time.time() > deadline:
            break
        sa_centers, sa_score = _optimize_with_sa(ls_centers, time_limit=10.0, seed=r+100)
        if sa_score > best_score:
            best_centers = sa_centers
            best_score = sa_score
            
    return best_centers
# EVOLVE-BLOCK-END
