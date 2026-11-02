import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = 145.0  # seconds, safety margin
    
    # 1. Generate candidate vectors with squared norm 4
    candidates = []
    
    # Type 1: One non-zero entry +/- 2
    for i in range(D):
        for s in [2, -2]:
            v = [0] * D
            v[i] = s
            candidates.append(v)
            
    # Type 2: Four non-zero entries +/- 1
    # Choose 4 indices from 11
    for idxs in itertools.combinations(range(D), 4):
        # 2^4 sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    N = len(candidates)
    
    # 2. Compute Dot Product Matrix
    M = candidates
    D_mat = M @ M.T
    
    # 3. Determine Conflicts
    # Conflict if dot product >= 3
    conflict_mask = (D_mat >= 3)
    np.fill_diagonal(conflict_mask, False)
    
    # Precompute degrees
    degrees = np.sum(conflict_mask, axis=1)
    
    best_set = np.array([], dtype=np.int64)
    best_size = 0
    
    # Heuristic: Sort indices by degree (ascending) to prioritize 'easy' nodes
    # We will use this order as a base, but introduce randomness or multiple passes.
    
    # To avoid O(N^2) in Python loops for the greedy step, we use vectorized operations
    # where possible, but the core MIS logic is sequential.
    # N=5300. 50 restarts * 5300 iterations * (vector check) 
    # Vector check: conflict_mask[v, current_set] is length ~size.
    # This might be slow in Python.
    
    # Optimization: Use a bitset or integer array for current_set conflicts?
    # conflict_mask is bool. 
    # Let's try to speed up the inner loop.
    
    # Alternative: Use a simpler greedy that is faster to implement in Python
    # by pre-computing neighbor lists? No, N^2 is small enough for matrix ops.
    
    # Let's run a few different greedy strategies.
    
    strategies = [
        "degree_asc",
        "degree_desc", 
        "random"
    ]
    
    for restart in range(100): # More restarts, but each is faster/shorter if we break early
        if time.time() > deadline - 5:
            break
            
        # Choose strategy
        if restart % 3 == 0:
            order = np.argsort(degrees) # Ascending degree
        elif restart % 3 == 1:
            order = np.argsort(-degrees) # Descending degree
        else:
            order = np.random.permutation(N)
            
        current_set = []
        in_set = np.zeros(N, dtype=bool)
        
        # Greedy Pass
        # To speed up, we can check conflicts against the *entire* set of candidates
        # that are already in the set.
        
        # Efficient check: 
        # If we maintain an array `blocked` of size N, where blocked[i] is True if i conflicts with any node in current_set.
        # When we add node v, we update blocked: blocked |= conflict_mask[v]
        # Then we just pick the next v from order where not in_set[v] and not blocked[v].
        
        blocked = np.zeros(N, dtype=bool)
        
        for v_idx in order:
            if in_set[v_idx] or blocked[v_idx]:
                continue
            
            # Add v_idx
            current_set.append(v_idx)
            in_set[v_idx] = True
            # Update blocked mask
            blocked |= conflict_mask[v_idx]
            
        current_set_arr = np.array(current_set, dtype=np.int64)
        
        # Local Search: Try to add any remaining vertex that is not blocked?
        # The greedy above with `blocked` update is actually a valid maximal independent set construction.
        # Because if a vertex is blocked, it conflicts with something in the set.
        # If it's not blocked, it doesn't conflict with anything in the set, so it could have been added.
        # So the resulting set is maximal.
        
        # To improve size, we can try swapping.
        # For simplicity and speed, we'll just take the best maximal set found.
        # Maximal IS is often close to Max IS.
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set_arr.copy()
            
    # Ensure we have at least one vector to avoid empty array error
    if len(best_set) == 0:
        # Fallback to a single valid vector
        best_set = np.array([0], dtype=np.int64)
        
    result_vectors = candidates[best_set]
    
    return result_vectors
