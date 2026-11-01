import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Generate candidate vectors with norm squared 4
    # Type A: 2 * e_i
    candidates = []
    for i in range(D):
        v = np.zeros(D, dtype=np.int64)
        v[i] = 2
        candidates.append(v)
        v[i] = -2
        candidates.append(v)
        
    # Type B: e_i +/- e_j +/- e_k +/- e_l
    # Indices i, j, k, l distinct
    indices = list(range(D))
    for combo in itertools.combinations(indices, 4):
        i, j, k, l = combo
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    for s4 in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        v[k] = s3
                        v[l] = s4
                        candidates.append(v)
                        
    candidates = np.array(candidates, dtype=np.int64)
    N = len(candidates)
    
    # 2. Compute dot product matrix
    # D_mat[i, j] = candidates[i] . candidates[j]
    D_mat = candidates @ candidates.T
    
    # 3. Greedy Independent Set
    # We want max subset S such that D_mat[i, j] <= 2 for all i != j in S
    # Conflict if D_mat[i, j] > 2
    
    best_set = []
    best_size = 0
    
    # We can run greedy multiple times with different orders
    # To save time, we'll run it a few times
    num_runs = 10
    rng = np.random.default_rng(42)
    
    for run in range(num_runs):
        if time.time() > deadline:
            break
            
        # Shuffle order of candidates
        order = np.arange(N)
        rng.shuffle(order)
        
        selected = []
        available = np.ones(N, dtype=bool)
        
        # Iterate through shuffled order
        for idx in order:
            if not available[idx]:
                continue
                
            # Select idx
            selected.append(idx)
            
            # Find conflicts with all currently available nodes
            # D_mat[idx, available] > 2
            conflicts = (D_mat[idx, available] > 2)
            
            # Mark conflicts as unavailable
            # available[conflicts] = False  -- This is tricky with boolean indexing on a view
            # We need to update the 'available' mask
            
            # Get indices of available nodes
            avail_indices = np.where(available)[0]
            # Check conflicts for these
            conflict_mask = D_mat[idx, avail_indices] > 2
            # Remove them from available
            available[avail_indices[conflict_mask]] = False
            available[idx] = False
            
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
            
    # Convert best_set indices to vectors
    if best_set:
        result = candidates[best_set]
    else:
        # Fallback to R^2=2 set if something went wrong
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        result = np.array(vectors, dtype=np.int64)
        
    return result
