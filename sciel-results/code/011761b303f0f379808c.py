import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate all 3-sparse vectors with entries +/- 1
    vectors = []
    indices = list(range(D))
    for comb in itertools.combinations(indices, 3):
        i, j, k = comb
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    vectors.append(v)
    
    vectors = np.array(vectors)
    N = len(vectors)
    
    # Precompute dot products? 
    # N=1320. N^2 = 1.7M. 
    # Dot product matrix is 1320x1320. 
    # This is small enough to compute directly or use matrix multiplication.
    
    # To speed up greedy, we can precompute the conflict matrix.
    # Conflict if dot > 1.
    
    # However, recalculating dot products for each candidate against all selected
    # is O(N * |selected|). 
    # If |selected| grows to ~800, and N=1320, total ops ~ 1M.
    # This is fast.
    
    # Let's use a greedy approach with multiple random restarts.
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_vectors = []
    best_size = 0
    
    # Heuristic: Sort vectors? 
    # Random permutation is good.
    
    # We can also try to be smarter.
    # But random restarts are robust.
    
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        perm = np.random.permutation(N)
        selected_idx = []
        
        # To make it faster, we can maintain a list of selected vectors
        # and check against them.
        
        for idx in perm:
            v = vectors[idx]
            # Check compatibility with all selected
            # v is (D,), selected is list of indices
            # We can use np.dot for batch if we store selected in array
            # But list is fine for small sizes.
            
            is_compatible = True
            for s_idx in selected_idx:
                dot = np.dot(v, vectors[s_idx])
                if dot > 1:
                    is_compatible = False
                    break
            
            if is_compatible:
                selected_idx.append(idx)
        
        if len(selected_idx) > best_size:
            best_size = len(selected_idx)
            best_vectors = [vectors[i] for i in selected_idx]
            
    return np.array(best_vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
