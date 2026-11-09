"""Program for no-isosceles set task using bitset-based distance tracking."""
import numpy as np
import random

N = 64

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a set of grid points in [0, N)^2 with no three points forming an isosceles triangle.
    
    Approach: 
    1. Precompute squared distance matrix for all N*N grid points.
    2. Use a randomized greedy algorithm with multiple restarts.
    3. For each candidate point, check validity using bitsets for distance conflicts.
    """
    N_local = N
    total_points = N_local * N_local
    
    # Precompute coordinates
    coords = np.array([(x, y) for x in range(N_local) for y in range(N_local)], dtype=np.int32)
    
    # Precompute squared distance matrix
    # D[i, j] = squared distance between point i and point j
    # We can compute this using broadcasting
    # coords shape: (M, 2)
    # diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (M, M, 2)
    # D = np.sum(diff**2, axis=2)
    
    # This creates a large intermediate array (4096*4096*2). 4096*4096*2*4 bytes = 128MB.
    # It might be faster to do it in chunks or just use loops if memory is an issue, but 128MB is usually fine.
    # Let's try to be safe and compute it in a way that doesn't blow up memory if possible, 
    # or just accept it.
    
    # Alternative: Compute D on the fly but cached? No, precomputation is better.
    
    # Let's use a loop to fill D to avoid the huge temporary array if possible, 
    # or just use the vectorized version.
    # Vectorized:
    dx = coords[:, 0:1] - coords[0:1, :]
    dy = coords[:, 1:2] - coords[1:2, :]
    D = dx**2 + dy**2
    D = D.astype(np.int32)
    
    # Greedy function
    def greedy_random(seed):
        rng = random.Random(seed)
        indices = list(range(total_points))
        rng.shuffle(indices)
        
        selected_indices = []
        dist_bitsets = [] # dist_bitsets[i] corresponds to selected_indices[i]
        
        for p_idx in indices:
            if not selected_indices:
                selected_indices.append(p_idx)
                dist_bitsets.append(0)
                continue
                
            # Get distances from p_idx to all selected points
            # D[p_idx, selected_indices]
            dists = D[p_idx, selected_indices]
            
            # Check 1: Distinctness
            # If any distance is repeated, p_idx is apex of isosceles triangle
            # Using a set for small arrays is fast
            if len(set(dists.tolist())) != len(dists):
                continue
            
            # Check 2: Conflict with existing distance bitsets
            # For each selected point s_idx, d(p, s) must not be in dist_bitsets[s_idx]
            valid = True
            for i, s_idx in enumerate(selected_indices):
                d2 = dists[i]
                # Check if bit d2 is set in dist_bitsets[i]
                if (dist_bitsets[i] >> d2) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # All checks passed, add point
            selected_indices.append(p_idx)
            
            # Build bitset for the new point
            new_bitset = 0
            for d2 in dists:
                new_bitset |= (1 << d2)
            dist_bitsets.append(new_bitset)
            
            # Update existing bitsets
            for i, d2 in enumerate(dists):
                dist_bitsets[i] |= (1 << d2)
                
        return selected_indices

    # Run multiple random restarts
    best_set = []
    num_trials = 50 # Adjust based on time limit. 300s is plenty for 50 trials of 4096 points.
    
    for t in range(num_trials):
        current_set = greedy_random(t)
        if len(current_set) > len(best_set):
            best_set = current_set
            # Optional: Early stop if we hit a known good value? No, keep trying.
            
    # Convert indices back to (x, y) coordinates
    result = [tuple(coords[idx]) for idx in best_set]
    return result
# EVOLVE-BLOCK-END
