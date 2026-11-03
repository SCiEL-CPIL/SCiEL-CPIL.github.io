import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    candidates = []
    indices = range(D)
    for combo in itertools.combinations(indices, 3):
        i, j, k = combo
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int8)
    n = len(candidates)
    
    # Calculate conflict degrees
    # Conflicts if dot >= 2
    degrees = np.zeros(n, dtype=np.int32)
    
    # This O(N^2) might be slow in Python if not vectorized.
    # 1320^2 = 1.7M dots.
    # We can vectorize the dot product calculation for the whole matrix?
    # Dot matrix = candidates @ candidates.T
    # 1320x1320 matrix of ints.
    # Memory: 1320*1320*4 bytes ~ 7MB. Fine.
    
    dot_products = candidates @ candidates.T
    # Diagonal is 3.
    # Conflict if dot >= 2.
    # Note: dot_products[i][j] == dot_products[j][i]
    
    # We only need to check j > i for degrees?
    # Or just count all j != i where dot >= 2.
    conflict_mask = dot_products >= 2
    np.fill_diagonal(conflict_mask, False)
    
    degrees = np.sum(conflict_mask, axis=1)
    
    best_set = []
    
    # Strategy 1: Greedy by degree (lowest first)
    order = np.argsort(degrees)
    current_set = []
    current_indices = set()
    for idx in order:
        v = candidates[idx]
        # Check compatibility
        # We can check against current_set
        # Or use the precomputed mask?
        # If we use mask, we need to check if any neighbor in current_set is in conflict.
        # neighbors = np.where(conflict_mask[idx])[0]
        # if any in current_indices -> conflict.
        
        # This is fast.
        neighbors = np.where(conflict_mask[idx])[0]
        # Check intersection with current_indices
        # set intersection is O(1) if we use sets?
        # But creating a set of neighbors each time is O(deg).
        # deg is ~55.
        
        if not current_indices.intersection(neighbors):
             current_set.append(idx)
             current_indices.add(idx)
             
    if len(current_set) > len(best_set):
        best_set = current_set[:]

    # Strategy 2: Random restarts with greedy
    for restart in range(50):
        if time.time() > deadline:
            break
        order = np.random.permutation(n)
        current_set = []
        current_indices = set()
        for idx in order:
            neighbors = np.where(conflict_mask[idx])[0]
            if not current_indices.intersection(neighbors):
                current_set.append(idx)
                current_indices.add(idx)
                
        if len(current_set) > len(best_set):
            best_set = current_set[:]

    if not best_set:
        # Fallback
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    result = candidates[best_set]
    return result.astype(np.int64)
