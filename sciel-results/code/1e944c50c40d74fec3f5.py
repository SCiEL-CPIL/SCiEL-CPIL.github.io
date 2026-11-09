"""
Rewritten program for 100 equal spheres in a unit cube; 
Uses targeted local search (best-neighbor moves) to maximize minimum distance.
"""
import numpy as np
import time
import sys

N = 100

def _compute_score(centres):
    """Compute the largest common radius for the given centres.
    Returns -inf if invalid."""
    if centres.shape != (N, 3):
        return -np.inf
    if not np.all(np.isfinite(centres)):
        return -np.inf
    # Check inside cube [-0.5, 0.5]^3
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return -np.inf
    # Min distance from boundary
    dist_boundary = 0.5 - np.max(np.abs(centres), axis=1)
    min_boundary = np.min(dist_boundary)
    if min_boundary < -1e-15:
        return -np.inf
    # Pairwise distances
    min_pair = np.inf
    for i in range(N):
        diffs = centres[i+1:] - centres[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        if len(dists) > 0:
            min_pair = min(min_pair, np.min(dists))
    if not np.isfinite(min_pair):
        return -np.inf
    return min(min_pair / 2.0, min_boundary)

def _cubic_grid(N):
    """Generate a cubic grid of N points."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    # Ensure we have at least N points
    # If m^3 < N, we need to adjust or just use the first N of a larger grid?
    # For N=100, m=5 (125 points). We take the first 100.
    # To be more symmetric, we can center the grid.
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    if len(grid) < N:
        # Fallback to random if grid is too small (should not happen for 100)
        return (np.random.rand(N, 3) - 0.5)
    return grid[:N].copy()

def _local_search(centres, time_limit=10.0, initial_step=0.01):
    """
    Local search algorithm that moves spheres to increase the minimum distance.
    Focuses on the 'bottleneck' constraints.
    """
    best_centres = centres.copy()
    best_score = _compute_score(centres)
    current_centres = centres.copy()
    current_score = best_score
    
    start_time = time.time()
    step = initial_step
    stagnant = 0
    max_stagnant = 500
    
    # Pre-allocate for performance
    # We will recompute distances each iteration for simplicity, 
    # but we can optimize by only checking neighbors if needed.
    # For N=100, full recomputation is O(N^2) = 10k ops, which is fast.
    
    while time.time() - start_time < time_limit:
        # Compute current score and identify bottleneck
        min_boundary = np.inf
        min_boundary_idx = -1
        min_pair_dist = np.inf
        min_pair_i = -1
        min_pair_j = -1
        
        # Boundary check
        dists_boundary = 0.5 - np.max(np.abs(current_centres), axis=1)
        min_boundary = np.min(dists_boundary)
        min_boundary_idx = np.argmin(dists_boundary)
        
        # Pairwise check
        # Vectorized min pair distance
        # Create upper triangular indices
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = current_centres[i_idx] - current_centres[j_idx]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        
        min_pair_idx = np.argmin(dists)
        min_pair_dist = dists[min_pair_idx]
        min_pair_i = i_idx[min_pair_idx]
        min_pair_j = j_idx[min_pair_idx]
        
        # Determine the global minimum constraint
        # Radius is min(min_pair_dist/2, min_boundary)
        # If min_pair_dist/2 < min_boundary, the bottleneck is the pair
        # Else, the bottleneck is the boundary
        
        if min_pair_dist / 2.0 < min_boundary:
            # Bottleneck is pair (i, j)
            i, j = min_pair_i, min_pair_j
            # Move i away from j, and j away from i
            diff = current_centres[i] - current_centres[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                dir_vec = diff / norm
                # Try moving
                new_i = current_centres[i] + dir_vec * step
                new_j = current_centres[j] - dir_vec * step
                
                # Clamp to box
                new_i = np.clip(new_i, -0.5, 0.5)
                new_j = np.clip(new_j, -0.5, 0.5)
                
                # Check if this move improves the minimum distance for these two
                # And doesn't make other distances worse (rough check)
                # For speed, we just accept if it increases the distance between i and j
                # and they stay in the box. The global score check will catch regressions.
                
                old_i = current_centres[i].copy()
                old_j = current_centres[j].copy()
                
                current_centres[i] = new_i
                current_centres[j] = new_j
                
                # Quick check: did the distance between i and j increase?
                new_dist = np.linalg.norm(current_centres[i] - current_centres[j])
                if new_dist <= min_pair_dist:
                    # Revert if no improvement
                    current_centres[i] = old_i
                    current_centres[j] = old_j
                else:
                    # Calculate new global score
                    new_score = _compute_score(current_centres)
                    if new_score > current_score:
                        current_score = new_score
                        stagnant = 0
                        if new_score > best_score:
                            best_score = new_score
                            best_centres = current_centres.copy()
                    else:
                        # Revert if global score didn't improve
                        current_centres[i] = old_i
                        current_centres[j] = old_j
                        stagnant += 1
            else:
                stagnant += 1
        else:
            # Bottleneck is boundary for index min_boundary_idx
            idx = min_boundary_idx
            # Move towards center
            # Direction is -current_centres[idx] (normalized)
            pos = current_centres[idx]
            norm = np.linalg.norm(pos)
            if norm > 1e-10:
                dir_vec = -pos / norm
                new_pos = pos + dir_vec * step
                new_pos = np.clip(new_pos, -0.5, 0.5)
                
                old_pos = pos.copy()
                current_centres[idx] = new_pos
                
                # Check if distance to boundary increased
                new_dist_boundary = 0.5 - np.max(np.abs(new_pos))
                if new_dist_boundary <= min_boundary:
                    current_centres[idx] = old_pos
                else:
                    new_score = _compute_score(current_centres)
                    if new_score > current_score:
                        current_score = new_score
                        stagnant = 0
                        if new_score > best_score:
                            best_score = new_score
                            best_centres = current_centres.copy()
                    else:
                        current_centres[idx] = old_pos
                        stagnant += 1
            else:
                stagnant += 1
                
        # Adapt step size
        if stagnant > max_stagnant:
            step *= 0.95
            stagnant = 0
            if step < 1e-7:
                break
            
    return best_centres, best_score

def _refine_with_kicks(centres, time_limit=10.0):
    """
    If local search gets stuck, try random kicks to escape local minima.
    """
    best_centres = centres.copy()
    best_score = _compute_score(centres)
    current_centres = centres.copy()
    
    start_time = time.time()
    kick_size = 0.05
    
    while time.time() - start_time < time_limit:
        # Perform a few local search steps
        c, s = _local_search(current_centres, time_limit=0.5, initial_step=0.005)
        if s > best_score:
            best_score = s
            best_centres = c.copy()
            current_centres = c.copy()
        
        # Apply a random kick to a random subset of points
        n_kicks = 5
        for _ in range(n_kicks):
            idx = np.random.randint(N)
            current_centres[idx] += np.random.normal(0, kick_size, size=3)
            current_centres[idx] = np.clip(current_centres[idx], -0.5, 0.5)
            
        # If the kick made the score much worse, revert to best
        curr_score = _compute_score(current_centres)
        if curr_score < best_score * 0.9:
            current_centres = best_centres.copy()
        else:
            # Continue from current state
            pass
            
        # Slowly reduce kick size
        kick_size *= 0.99
        
    return best_centres, best_score

def construct():
    """Main entry point: search for best packing of 100 spheres in unit cube."""
    deadline = time.time() + 140.0  # Leave 10s margin
    
    # 1. Initial Grid
    centres = _cubic_grid(N)
    best_centres = centres.copy()
    best_score = _compute_score(centres)
    
    # 2. Initial Local Search
    c, s = _local_search(centres, time_limit=20.0, initial_step=0.02)
    if s > best_score:
        best_score = s
        best_centres = c.copy()
        
    # 3. Refinement with smaller steps
    if time.time() < deadline - 30:
        c, s = _local_search(best_centres, time_limit=30.0, initial_step=0.005)
        if s > best_score:
            best_score = s
            best_centres = c.copy()
            
    # 4. Kick-and-Refine to escape local minima
    if time.time() < deadline - 10:
        c, s = _refine_with_kicks(best_centres, time_limit=20.0)
        if s > best_score:
            best_score = s
            best_centres = c.copy()
            
    # 5. Final Polish
    if time.time() < deadline - 2:
        c, s = _local_search(best_centres, time_limit=5.0, initial_step=0.001)
        if s > best_score:
            best_score = s
            best_centres = c.copy()
            
    # Ensure validity
    final_score = _compute_score(best_centres)
    if final_score <= 0 or not np.isfinite(final_score):
        # Fallback
        best_centres = _cubic_grid(N)
        
    return best_centres
