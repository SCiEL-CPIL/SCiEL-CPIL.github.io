import itertools
import numpy as np
import random
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate all weight-3 vectors with entries +/- 1
    # Support is a combination of 3 indices from 11
    # Signs are 3 bits
    vectors = []
    indices = []
    for support in itertools.combinations(range(D), 3):
        i, j, k = support
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    v[k] = s3
                    vectors.append(v)
                    
    n_vecs = len(vectors)
    
    # Precompute conflict matrix
    # Conflict if dot product >= 2
    # v . w >= 2
    # Since entries are +/-1 or 0, dot is integer.
    # For weight 3, max dot is 3 (identical).
    # We want to avoid dot >= 2.
    
    # Efficient dot product calculation
    # v and w are lists of ints.
    # We can use numpy for speed if we convert to array, but list ops might be fine for 1320.
    # Let's use numpy for the matrix.
    V = np.array(vectors, dtype=np.int8)
    
    # Dot product matrix
    # M[i, j] = V[i] . V[j]
    M = V @ V.T
    
    # Conflict if M[i, j] >= 2
    # Note: M[i, i] = 3. We don't select same vector twice.
    
    conflict = M >= 2
    
    best_count = 0
    best_indices = []
    
    # Greedy with random restarts
    # We can do as many restarts as time allows
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        order = rng.permutation(n_vecs)
        selected = np.zeros(n_vecs, dtype=bool)
        count = 0
        
        for idx in order:
            if time.time() > deadline:
                break
            # Check if idx conflicts with any selected
            # conflict[idx, selected]
            # If any is True, skip.
            if not np.any(conflict[idx, selected]):
                selected[idx] = True
                count += 1
                
        if count > best_count:
            best_count = count
            best_indices = np.where(selected)[0]
            
    # Construct result
    if best_count == 0:
        # Fallback to seed-like behavior if something went wrong (should not happen)
        # Return weight 2 vectors
        vectors_w2 = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors_w2.append(v)
        return np.array(vectors_w2, dtype=np.int64)
        
    result = V[best_indices]
    return result.astype(np.int64)
# EVOLVE-BLOCK-END
