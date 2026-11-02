"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def _generate_candidates(weights):
    """Generate all integer vectors with the specified weights."""
    candidates = []
    for w in weights:
        if w > D:
            continue
        # Choose w indices out of D
        for indices in itertools.combinations(range(D), w):
            # Choose signs for these indices
            for signs in itertools.product([1, -1], repeat=w):
                v = [0] * D
                for idx, sign in zip(indices, signs):
                    v[idx] = sign
                candidates.append(v)
    return np.array(candidates, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Generate candidates with weight 3 and 4
    candidates = _generate_candidates([3, 4])
    n_cand = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Precompute conflict matrix
    # Conflict if dist_sq < max(norm_i, norm_j)
    # dist_sq = n_i + n_j - 2 dot(i, j)
    # Condition: n_i + n_j - 2 dot(i, j) < max(n_i, n_j)
    # Let's compute dot products matrix
    dots = candidates @ candidates.T
    
    # Max norms matrix
    max_norms = np.maximum(norms_sq[:, np.newaxis], norms_sq[np.newaxis, :])
    
    # Dist squared matrix
    dist_sq = norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - 2 * dots
    
    # Conflict mask: True if incompatible
    conflict = dist_sq < max_norms
    
    # Diagonal is always compatible with itself, but we don't pick same vector twice
    # We want to find a large subset of indices such that no pair conflicts
    
    best_set = set()
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    # Strategy: Greedy with random restarts using the precomputed conflict matrix
    # Optimization: Represent the current selected set as a boolean mask or list
    # To speed up, we can maintain a 'available' mask and update it
    
    max_restarts = 500
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random permutation of candidates
        perm = rng.permutation(n_cand)
        
        selected = np.zeros(n_cand, dtype=bool)
        available = np.ones(n_cand, dtype=bool)
        count = 0
        
        for idx in perm:
            if not available[idx]:
                continue
            
            # Check if idx is compatible with all currently selected
            # We can pre-compute the conflict rows for selected items?
            # Or just check conflict[idx, selected]
            
            if np.any(conflict[idx, selected]):
                continue
            
            # Select idx
            selected[idx] = True
            count += 1
            
            # Update available: remove idx and all its neighbors
            available[idx] = False
            # Neighbors are those j where conflict[idx, j] is True
            # available[conflict[idx, :]] = False  -- this is O(N) per selection
            # This is the bottleneck. 
            # Alternative: Just check compatibility on the fly against selected list.
            # But selected list grows.
            
            # Let's stick to the on-the-fly check against selected indices for simplicity 
            # but optimize the inner loop.
            # Actually, the "available" mask update is O(N). 
            # With N=6600 and ~500 selections, that's 3.3M ops per restart.
            # 500 restarts -> 1.65B ops. Might be too slow in Python.
            
            # Let's try a different heuristic: 
            # Greedy by "least constrained" (fewest conflicts with current set)
            # This requires calculating conflicts for all available items at each step.
            # Too slow.
            
            # Let's go back to simple random greedy but vectorized check against selected.
            # The previous code did this. Why was it slow?
            # It converted selected_vectors to array every time.
            # We can maintain a list of selected indices and use fancy indexing on the conflict matrix.
            
            pass # This branch is just thinking. I will implement the vectorized check against selected indices.

        # The loop above was a draft. Let's implement the real logic.
        
    # Re-implementing the loop properly
    
    best_vectors = []
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        perm = rng.permutation(n_cand)
        selected_indices = []
        
        for idx in perm:
            if not selected_indices:
                selected_indices.append(idx)
                continue
            
            # Check compatibility with all selected
            # conflict[idx, selected_indices]
            # If any is True, skip
            if np.any(conflict[idx, selected_indices]):
                continue
            
            selected_indices.append(idx)
            
        if len(selected_indices) > best_count:
            best_count = len(selected_indices)
            best_vectors = selected_indices.copy()
            
            # If we hit a very high number, we can break
            if best_count > 1500:
                break

    if not best_vectors:
        # Fallback
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    return candidates[best_vectors]

# EVOLVE-BLOCK-END
