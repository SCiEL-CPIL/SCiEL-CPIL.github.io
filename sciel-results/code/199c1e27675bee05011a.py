# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

N = 50

def _get_min_dist(centers):
    """
    Compute the minimum distance between any two centers and the minimum distance to the wall.
    Returns (min_pair_dist, min_wall_dist).
    """
    # Min wall dist
    # Cube is [-0.5, 0.5]^3
    # dist to wall = 0.5 - max(|x|, |y|, |z|) is NOT correct for min over all walls.
    # dist to nearest wall for point c is min(0.5-c[0], 0.5+c[0], ...)
    # This is equivalent to 0.5 - max(|c[0]|, |c[1]|, |c[2]|) ? No.
    # Let c = (0.4, 0, 0). dist to right wall = 0.1. dist to left = 0.9. min = 0.1.
    # max(|0.4|, 0, 0) = 0.4. 0.5 - 0.4 = 0.1. Correct.
    # Let c = (0.4, 0.4, 0). dist to right = 0.1, dist to front = 0.1. min = 0.1.
    # max(0.4, 0.4, 0) = 0.4. 0.5 - 0.4 = 0.1. Correct.
    # So min_wall_dist = 0.5 - max(np.abs(centers), axis=1).min()
    
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    
    # Min pair dist using KDTree
    if len(centers) < 2:
        return float('inf'), min_wall
        
    tree = cKDTree(centers)
    # query_pairs returns pairs of points with distance <= r
    # We want the smallest distance.
    # We can query k=2 to get nearest neighbor for each point.
    # But that's O(N log N). 
    # query_pairs with r=inf is bad.
    # Let's just get the nearest neighbor for each point.
    dists, idxs = tree.query(centers, k=2)
    # dists[:, 1] is the distance to the nearest neighbor.
    min_pair = np.min(dists[:, 1])
    
    return min_pair, min_wall

def _score(centers):
    """
    Score is the radius r such that spheres of radius r fit.
    r <= min_pair_dist / 2
    r <= min_wall_dist
    So score = min(min_pair_dist / 2, min_wall_dist)
    """
    min_pair, min_wall = _get_min_dist(centers)
    return min(min_pair / 2.0, min_wall)

def _force_directed_search(centers, time_limit, seed=42):
    """
    Iteratively move points to increase the minimum distance.
    Uses a repulsion model: if points are too close, push them apart.
    """
    rng = np.random.RandomState(seed)
    current = centers.copy()
    best = current.copy()
    best_score = _score(best)
    
    # We want to maximize the minimum distance.
    # Let current_min_pair, current_min_wall = _get_min_dist(current)
    # Target radius r is best_score.
    
    # We will iterate, moving points that are "constrained" (part of the minimum distance).
    
    # Adaptive step size
    step_size = 0.01
    no_improve_count = 0
    max_no_improve = 1000
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        min_pair, min_wall = _get_min_dist(current)
        current_score = min(min_pair / 2.0, min_wall)
        
        if current_score > best_score:
            best_score = current_score
            best = current.copy()
            no_improve_count = 0
            # Slightly increase step size if we are improving? Or keep it.
            # step_size *= 1.05 
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break

        # Identify points to move
        # If min_pair is the bottleneck:
        if min_pair / 2.0 < min_wall:
            # Find pairs with distance close to min_pair
            # Using KDTree to find neighbors within a small epsilon of min_pair
            # Or just find the specific pair(s) causing the minimum.
            # For efficiency, let's just pick a random point involved in a close pair.
            # To be faster, we can approximate: pick a random point, check its neighbors.
            
            # Let's use a simpler heuristic:
            # Pick a random point i.
            # Find its nearest neighbor j.
            # If dist(i,j) is close to min_pair, push i away from j.
            
            idx = rng.randint(N)
            # Get neighbors
            # We need the nearest neighbor of idx
            # We can re-query the tree or store it? 
            # Re-querying is O(log N).
            tree = cKDTree(current)
            dists, idxs = tree.query(current, k=2)
            
            # For each point, we have its nearest neighbor distance.
            # We want to move points that have small nearest neighbor distances.
            # Let's sort points by their nearest neighbor distance and pick from the worst ones.
            nn_dists = dists[:, 1]
            
            # Threshold: move if nn_dist < min_pair * 1.1
            threshold = min_pair * 1.05
            candidates = np.where(nn_dists < threshold)[0]
            
            if len(candidates) == 0:
                # If no obvious candidates, pick random
                idx = rng.randint(N)
                nn_idx = idxs[idx, 1]
                dist = nn_dists[idx]
                if dist > 1e-9:
                    direction = (current[idx] - current[nn_idx]) / dist
                    # Move away
                    current[idx] += direction * step_size
                else:
                    current[idx] += rng.normal(0, step_size, 3)
            else:
                # Pick one from candidates
                idx = candidates[rng.randint(len(candidates))]
                nn_idx = idxs[idx, 1]
                dist = nn_dists[idx]
                if dist > 1e-9:
                    direction = (current[idx] - current[nn_idx]) / dist
                    # Move away
                    current[idx] += direction * step_size
                else:
                    current[idx] += rng.normal(0, step_size, 3)
            
        else:
            # Wall is the bottleneck
            # Find points closest to the wall
            wall_dists = 0.5 - np.max(np.abs(current), axis=1)
            min_wall_val = np.min(wall_dists)
            threshold = min_wall_val + 0.005
            candidates = np.where(wall_dists < threshold)[0]
            
            if len(candidates) == 0:
                idx = rng.randint(N)
            else:
                idx = candidates[rng.randint(len(candidates))]
            
            # Move towards center (origin)
            # Direction is -current[idx]
            # Normalize
            norm = np.linalg.norm(current[idx])
            if norm > 1e-9:
                direction = -current[idx] / norm
            else:
                direction = rng.normal(0, 1, 3)
                direction /= np.linalg.norm(direction)
            
            current[idx] += direction * step_size
            
        # Clamp to cube
        current = np.clip(current, -0.5, 0.5)
        
        # Decay step size if we are stuck
        if no_improve_count > 100:
            step_size *= 0.99
            if step_size < 1e-5:
                # Reset step size and perturb?
                step_size = 0.01
                # Small random perturbation to escape local minima
                current += rng.normal(0, 0.005, size=(N, 3))
                current = np.clip(current, -0.5, 0.5)
                
    return best, best_score

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin for 150s limit
    
    # Initialization: Grid with perturbation
    # 50 points. 4x4x4 grid has 64. 
    # Let's use a 4x4x4 grid and pick 50.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Shuffle to break symmetry?
    # Or just take first 50.
    # Better: random subset of grid points?
    # Let's start with a perturbed grid to break degeneracies.
    
    rng = np.random.RandomState(42)
    centers = grid[:N].copy()
    centers += rng.normal(0, 0.02, size=(N, 3))
    centers = np.clip(centers, -0.5, 0.5)
    
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    # Run force directed search with time budget
    time_budget = deadline - time.time()
    if time_budget > 5:
        # Run for most of the time
        fd_centers, fd_score = _force_directed_search(centers, time_budget - 5, seed=42)
        if fd_score > best_score:
            best_centers = fd_centers
            best_score = fd_score
            
    # If we have time left, try another restart with different seed/perturbation
    if time.time() < deadline - 5:
        centers2 = best_centers + rng.normal(0, 0.01, size=(N, 3))
        centers2 = np.clip(centers2, -0.5, 0.5)
        time_budget2 = deadline - time.time() - 2
        if time_budget2 > 2:
            fd_centers2, fd_score2 = _force_directed_search(centers2, time_budget2, seed=123)
            if fd_score2 > best_score:
                best_centers = fd_centers2
                best_score = fd_score2

    return best_centers
# EVOLVE-BLOCK-END
