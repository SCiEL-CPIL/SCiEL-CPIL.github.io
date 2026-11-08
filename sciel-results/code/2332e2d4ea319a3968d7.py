"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    # Safety margin: 110s
    deadline = start_time + 110.0
    
    # 1. Generate Weight-4 Vectors
    # Norm squared R = 4. Condition: u.v <= 2.
    vectors_list = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(support, signs):
                v[idx] = val
            vectors_list.append(v)
            
    vectors_np = np.array(vectors_list, dtype=np.int32)
    N = len(vectors_np)
    
    # Fallback: Weight-2 vectors (Seed behavior)
    def get_fallback():
        fb_vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    fb_vectors.append(v)
        return np.array(fb_vectors, dtype=np.int64)

    if N == 0:
        return get_fallback()

    # 2. Compute Conflict Graph
    # Dot products
    if time.time() > deadline:
        return get_fallback()
        
    dot_matrix = vectors_np @ vectors_np.T
    
    # Conflict if dot > 2
    conflicts = dot_matrix > 2
    np.fill_diagonal(conflicts, False)
    
    # 3. Greedy Independent Set
    # Sort by degree (number of conflicts)
    degrees = np.sum(conflicts, axis=1)
    sorted_indices = np.argsort(degrees)
    
    blocked = np.zeros(N, dtype=bool)
    selected_indices = []
    
    for idx in sorted_indices:
        if time.time() > deadline:
            break
            
        if not blocked[idx]:
            selected_indices.append(idx)
            blocked[idx] = True
            # Block neighbors
            neighbors = np.where(conflicts[idx])[0]
            blocked[neighbors] = True
            
    if not selected_indices:
        return get_fallback()
        
    result = vectors_np[selected_indices].astype(np.int64)
    return result

# EVOLVE-BLOCK-END
