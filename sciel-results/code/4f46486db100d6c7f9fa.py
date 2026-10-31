import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0
    
    # Generate all integer vectors of norm 2 in R^11
    # Norm 2 means 4 non-zero entries, each +/-1
    indices = list(itertools.combinations(range(D), 4))
    signs = list(itertools.product([1, -1], repeat=4))
    
    candidates = []
    for idx in indices:
        for s in signs:
            v = np.zeros(D, dtype=np.int64)
            for k, val in zip(idx, s):
                v[k] = val
            candidates.append(v)
            
    M = np.array(candidates)
    N = M.shape[0]
    
    # Compute dot product matrix
    # D_mat[i, j] = dot(M[i], M[j])
    D_mat = M @ M.T
    
    best_set = []
    best_size = 0
    
    # Greedy search with random restarts
    # We aim to find a large subset of indices such that D_mat[i, j] <= 2 for all pairs
    while time.time() - start_time < deadline - 5.0:
        perm = np.random.permutation(N)
        candidates_arr = perm.copy()
        selected = []
        
        while len(candidates_arr) > 0:
            idx_pos = np.random.randint(len(candidates_arr))
            i = candidates_arr[idx_pos]
            
            # Remove i from candidates_arr
            # Swap with last element and pop
            candidates_arr[idx_pos] = candidates_arr[-1]
            candidates_arr = candidates_arr[:-1]
            
            selected.append(i)
            
            # Filter candidates_arr to keep only those compatible with i
            if len(candidates_arr) > 0:
                dots = D_mat[i, candidates_arr]
                mask = dots <= 2
                candidates_arr = candidates_arr[mask]
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
            
    if not best_set:
        # Fallback: return empty or seed
        # But this shouldn't happen
        return np.array([], dtype=np.int64)
        
    return M[np.array(best_set)]
# EVOLVE-BLOCK-END
