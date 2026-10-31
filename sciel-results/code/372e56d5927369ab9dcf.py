import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Generate all candidate vectors with 3 non-zero coordinates (+/- 1)
    candidates = []
    for idxs in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for i, s in zip(idxs, signs):
                v[i] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n_cands = len(candidates)
    
    best_k = 0
    best_vectors = np.empty((0, D), dtype=np.int64)
    
    # 2. Greedy search with multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Random permutation of candidates
        order = np.random.permutation(n_cands)
        
        selected_list = []
        # To speed up dot products, we can keep a NumPy array of selected vectors
        # but updating it is costly. 
        # Given the small dimension (11), simple list might be okay if we batch?
        # No, we need incremental check.
        
        # Optimization: 
        # If we just use the list, and convert to array for dot product:
        # The cost is O(N_selected).
        
        for idx in order:
            if time.time() > deadline:
                break
                
            u = candidates[idx]
            
            if not selected_list:
                selected_list.append(u)
                continue
                
            # Check compatibility
            # Convert to array for vectorized dot product
            # This is the bottleneck. 
            # Alternative: Keep a running sum? No, need max dot product.
            
            # Let's try to keep selected as a list of numpy arrays?
            # Or just a list of lists.
            
            # If N_selected is ~500, np.array(selected_list) takes time.
            # Let's estimate if 50 restarts * 1320 candidates * (conversion + dot) fits in 140s.
            
            sel_arr = np.array(selected_list, dtype=np.int64)
            dots = sel_arr @ u
            if np.any(dots > 1):
                continue
            else:
                selected_list.append(u)
                
        k = len(selected_list)
        if k > best_k:
            best_k = k
            best_vectors = np.array(selected_list, dtype=np.int64)
            
    return best_vectors
