import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate all integer vectors in R^11 with exactly four non-zero entries, each +/- 1
    # These have norm sqrt(4) = 2.
    # Condition: min pairwise distance >= max norm (2).
    # dist^2 = |u-v|^2 = |u|^2 + |v|^2 - 2 u.v = 4 + 4 - 2 u.v = 8 - 2 u.v
    # We need dist >= 2 => dist^2 >= 4 => 8 - 2 u.v >= 4 => 2 u.v <= 4 => u.v <= 2.
    # So we seek a subset where max dot product is <= 2.
    
    candidates = []
    indices = list(range(D))
    import itertools
    
    # Generate combinations of 4 indices
    for comb in itertools.combinations(indices, 4):
        # Generate all sign combinations (2^4 = 16)
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            candidates.append(v)
            
    candidates = np.array(candidates)
    n_candidates = len(candidates)
    
    # Precompute dot product matrix?
    # 5280 x 5280 is ~28M entries. int8 is enough (range -4 to 4).
    # This takes ~28MB memory. Very fast to compute with BLAS.
    dot_products = candidates @ candidates.T
    
    # Conflict matrix: True if dot product > 2 (i.e., 3 or 4)
    # Note: diagonal is 4 (self dot product). We ignore self.
    conflicts = dot_products > 2
    
    best_set = []
    best_size = 0
    
    # Randomized Greedy with restarts
    # We will run multiple passes.
    # In each pass, we shuffle candidates and greedily pick non-conflicting ones.
    
    # To make it faster, we can use a "coloring" or "independent set" heuristic.
    # Simple greedy:
    # 1. Shuffle indices
    # 2. For each idx in shuffled:
    #    if not conflict with any in current_set: add to current_set
    
    # Optimization: 
    # Instead of checking against the whole set every time, we can just check.
    # Since n_candidates is small, we can afford a few hundred restarts.
    
    rng = np.random.default_rng(42)
    
    for i in range(500): # Try 500 restarts
        if time.time() > deadline:
            break
            
        order = rng.permutation(n_candidates)
        current_set = []
        selected_mask = np.zeros(n_candidates, dtype=bool)
        
        for idx in order:
            # Check if candidate idx conflicts with any in current_set
            # We can check against the entire candidates array using selected_mask
            # conflicts[idx, selected_mask]
            if not np.any(conflicts[idx, selected_mask]):
                current_set.append(idx)
                selected_mask[idx] = True
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    # Return the best set found
    if not best_set:
        # Fallback to seed if something went wrong
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return candidates[best_set]
