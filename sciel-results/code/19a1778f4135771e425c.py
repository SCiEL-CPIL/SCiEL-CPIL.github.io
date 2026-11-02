import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate all D_11 vectors with squared norm 4
    candidates = []
    
    # Type 1: One ±2, rest 0
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four ±1's, rest 0
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict: M[i,j] > 2
    # We want an independent set in the graph of conflicts.
    # compatible[i,j] is True if M[i,j] <= 2
    compat = (M <= 2).astype(np.uint8)
    np.fill_diagonal(compat, 1) # Self is compatible with self for logic, but we don't pick same index twice
    
    # Separate weight-2 and weight-4 indices
    # Weight-2 vectors are the first 22
    w2_indices = list(range(22))
    w4_indices = list(range(22, N))
    
    # It is always optimal to include all weight-2 vectors
    # because they are compatible with all weight-4 vectors (dot product <= 2)
    # and with each other (dot product 0 or 4? 
    # v1=(2,0..), v2=(-2,0..) -> dot -4. OK.
    # v1=(2,0..), v3=(2,0..) -> same vector, not in list twice.
    # v1=(2,0..), v4=(0,2..) -> dot 0. OK.
    # So all weight-2 are mutually compatible.
    
    best_set = w2_indices[:]
    best_size = len(best_set)
    
    # Now find max independent set in weight-4 subgraph
    # Subgraph nodes: w4_indices
    # Compat matrix for w4: compat[np.ix_(w4_indices, w4_indices)]
    w4_compat = compat[np.ix_(w4_indices, w4_indices)]
    n_w4 = len(w4_indices)
    
    # Map local index to global index
    # local i corresponds to global w4_indices[i]
    
    # Greedy with random restarts
    rng = np.random.default_rng(42)
    
    # Precompute degrees for ordering
    degrees = (1 - w4_compat).sum(axis=1)
    
    best_w4_set = []
    
    max_restarts = 500
    restarts_done = 0
    
    while restarts_done < max_restarts and time.time() < deadline - 10:
        # Choose an ordering
        if restarts_done % 10 == 0:
            # Degree based
            order = np.argsort(degrees)
        else:
            # Random
            order = rng.permutation(n_w4)
        
        selected_local = []
        selected_mask = np.zeros(n_w4, dtype=bool)
        
        # Vectorized greedy
        # Instead of checking each one against all selected, we can maintain a mask of "available" nodes
        # But simple loop is O(N^2). With N=5280, N^2 is 27M ops.
        # In Python, a loop of 5280 iterations is fast.
        # The check `np.all(w4_compat[idx, selected_mask])` is the bottleneck if done naively.
        
        # Optimized Greedy:
        # Maintain a boolean array `available` which is initially all True.
        # When we pick a node, we AND `available` with `w4_compat[node, :]`.
        available = np.ones(n_w4, dtype=bool)
        
        for idx in order:
            if not available[idx]:
                continue
            
            # Pick idx
            selected_local.append(idx)
            available[idx] = False # Don't pick again
            
            # Update available: must be compatible with idx
            # available = available & w4_compat[idx, :]
            # This is a vector operation of size 5280.
            # Doing this 5280 times is 5280 * 5280 = 27M operations.
            # NumPy can do this in ~10-20ms per restart?
            # 500 restarts * 20ms = 10s. Feasible.
            
            available = available & w4_compat[idx, :]
        
        if len(selected_local) > len(best_w4_set):
            best_w4_set = selected_local[:]
            best_size = len(best_set) + len(best_w4_set)
        
        restarts_done += 1
        
    # Convert best_w4_set local indices to global
    global_w4_set = [w4_indices[i] for i in best_w4_set]
    final_set = best_set + global_w4_set
    
    # Try to expand final_set
    if time.time() < deadline - 5:
        in_final = np.zeros(N, dtype=bool)
        in_final[final_set] = True
        
        # Try to add any remaining vector
        for idx in range(N):
            if in_final[idx]:
                continue
            # Check if compatible with all in final_set
            # M[idx, final_set] <= 2
            if np.all(M[idx, final_set] <= 2):
                final_set.append(idx)
                in_final[idx] = True
        
        best_size = len(final_set)

    result = candidates[final_set].astype(np.int64)
    
    # Verify validity
    n = len(result)
    if n == 0:
        return candidates[:22].astype(np.int64)
    
    max_norm_sq = int(np.max(np.sum(result**2, axis=1)))
    # Vectorized check
    diffs = result[:, np.newaxis, :] - result[np.newaxis, :, :]
    diff_norms_sq = np.sum(diffs**2, axis=2)
    if np.min(diff_norms_sq[np.triu_indices(n, k=1)]) < max_norm_sq:
        # Invalid, fallback to just weight-2
        return candidates[:22].astype(np.int64)
    
    return result
