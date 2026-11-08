import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    # Safety margin: 10 seconds
    deadline = start_time + 110.0
    
    best_vectors = None
    best_count = 0
    
    # We focus on weight 4 as it provides the largest pool (5280 vectors)
    # and reasonable constraints (dot <= 2).
    
    # Generate all weight 4 vectors
    candidates = []
    indices = range(D)
    for comb in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(comb, signs):
                v[idx] = val
            candidates.append(v)
    
    V = np.array(candidates, dtype=np.int8)
    N = V.shape[0]
    
    if N == 0:
        # Fallback to weight 2 if something goes wrong (shouldn't happen)
        cands2 = []
        for comb in itertools.combinations(indices, 2):
            for signs in itertools.product([1, -1], repeat=2):
                v = [0] * D
                for idx, val in zip(comb, signs):
                    v[idx] = val
                cands2.append(v)
        return np.array(cands2, dtype=np.int64)

    # Precompute dot products
    # V is (N, D). V @ V.T is (N, N).
    # Max dot product is 4. int8 is enough for V, but dot product needs int16 or int32 to be safe?
    # 4*4=16. int8 max is 127. So int8 is fine for V, and result of V@V.T with int8 inputs 
    # in numpy might be int8? No, usually promotes. Let's use int32 for M to be safe and fast.
    M = V @ V.T
    M = M.astype(np.int32)
    
    # Conflict condition: dot > 2
    # We want to keep nodes where dot <= 2.
    # So compatible mask for idx is M[idx] <= 2.
    
    best_indices = None
    best_count = 0
    
    # Greedy Independent Set with random restarts
    # We use a vectorized approach for the active set.
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        # Random permutation
        perm = np.random.permutation(N)
        
        active = np.ones(N, dtype=bool)
        selected = []
        
        # Iterate through permuted indices
        for idx in perm:
            if active[idx]:
                selected.append(idx)
                # Update active set: remove idx and all its conflicts
                # Conflicts are where M[idx] > 2.
                # So we keep where M[idx] <= 2.
                active &= (M[idx] <= 2)
                # Ensure idx itself is marked inactive (though M[idx,idx]=4 > 2, so it's already excluded)
                # active[idx] will be False because M[idx, idx] = 4 > 2.
                
        if len(selected) > best_count:
            best_count = len(selected)
            best_indices = np.array(selected, dtype=int)
            
            # Early exit if we reach a very high number?
            # The theoretical max is not known, but let's see.
            # If we find 1000+, it's good.
            
    # If no indices found (should not happen), fallback
    if best_indices is None or len(best_indices) == 0:
        best_indices = np.array([0], dtype=int)
        
    result = V[best_indices]
    return result.astype(np.int64)
