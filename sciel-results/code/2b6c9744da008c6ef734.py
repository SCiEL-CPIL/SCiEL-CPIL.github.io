"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def construct():
    start_time = time.time()
    # Allow 110 seconds for the search to be safe
    deadline = start_time + 110.0

    # Generate Type A vectors: Norm squared 4, two non-zeros (one is +/-2)
    # Actually, (2,0,..) has norm 2.
    # (1,1,0,..) has norm sqrt(2).
    # Let's stick to Norm 2 (squared 4).
    # Type A: (2, 0, ..., 0) and (-2, 0, ..., 0)
    vectors_A = []
    for i in range(D):
        v = np.zeros(D, dtype=np.int64)
        v[i] = 2
        vectors_A.append(v)
        v[i] = -2
        vectors_A.append(v)
        
    # Type B: (1, 1, 1, 1, 0, ..., 0) with signs
    vectors_B = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(combo, signs):
                v[idx] = s
            vectors_B.append(v)
            
    vectors_A = np.array(vectors_A, dtype=np.int64)
    vectors_B = np.array(vectors_B, dtype=np.int64)
    
    n_B = len(vectors_B)
    
    # Precompute dot products for B
    dots = vectors_B @ vectors_B.T
    # Condition: dot <= 2
    incompat = dots > 2.0
    np.fill_diagonal(incompat, False)
    
    best_subset = np.array([], dtype=np.int64)
    best_size = 0
    
    # Randomized Greedy Search
    rng = np.random.default_rng(42)
    
    # We can run as many restarts as possible within the deadline
    while time.time() < deadline:
        order = rng.permutation(n_B)
        in_set = np.zeros(n_B, dtype=bool)
        current_size = 0
        
        for idx in order:
            if in_set[idx]:
                continue
            # Check compatibility
            # If idx is incompatible with any already in set, skip
            # in_set & incompat[idx] gives intersection
            if not np.any(in_set & incompat[idx]):
                in_set[idx] = True
                current_size += 1
                
        if current_size > best_size:
            best_size = current_size
            best_subset = np.where(in_set)[0]
            
    # Combine results
    if best_size == 0:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    selected_B = vectors_B[best_subset]
    all_vectors = np.vstack((vectors_A, selected_B))
    
    return all_vectors
# EVOLVE-BLOCK-END
