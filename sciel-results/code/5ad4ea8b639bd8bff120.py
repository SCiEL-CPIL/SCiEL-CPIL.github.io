import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Generate candidate vectors
    # Norm squared 2: 220 vectors
    # Norm squared 4: 
    #   Type 1: (2, 0, ...) -> 11 * 2 = 22 vectors
    #   Type 2: (1, 1, 1, 1, ...) -> 4 non-zeros. 
    #            Choose 4 indices: C(11,4) = 330.
    #            Signs: 2^4 = 16.
    #            Total: 330 * 16 = 5280.
    #   Type 3: (2, 1, 1, ...) -> 3 non-zeros? No, 4+1+1=6.
    #   Type 4: (1, 1, 1, 1) is norm 4.
    #   Type 5: (2, 0, 0, 0, 1, 1, ...) -> 2^2 + 1 + 1 = 6.
    
    # We'll generate all integer vectors with norm squared <= 10
    # to have a good pool.
    
    candidates = []
    
    # Norm squared 2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.append(np.array(v, dtype=np.int64))
                
    # Norm squared 4
    # (2, 0, ...)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            candidates.append(np.array(v, dtype=np.int64))
            
    # (1, 1, 1, 1, 0, ...)
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(indices, signs):
                v[idx] = s
            candidates.append(np.array(v, dtype=np.int64))
            
    # (2, 1, 1, 0, ...) -> 4 + 1 + 1 = 6 (skip for now, higher norm)
    # (1, 1, 1, 1, 1, 1, 1, 1) -> norm 8
    
    # Remove duplicates (shouldn't be any, but just in case)
    # Convert to set of tuples for uniqueness
    unique_candidates = []
    seen = set()
    for v in candidates:
        t = tuple(v)
        if t not in seen:
            seen.add(t)
            unique_candidates.append(v)
            
    # Sort by norm squared
    unique_candidates.sort(key=lambda v: np.dot(v, v))
    
    # Greedy selection
    selected = []
    max_norm_sq = 0
    
    # We start with the 220 norm-2 vectors as a strong base
    # Actually, let's just run the greedy on the sorted list.
    # But greedy on sorted list might pick too many short vectors and block longer ones?
    # No, shorter vectors are always better for packing.
    # The constraint is d_min >= max_norm.
    # If we pick short vectors, max_norm is small, so d_min requirement is small.
    # So we should prioritize short vectors.
    
    for v in unique_candidates:
        if time.time() > deadline:
            break
            
        v_norm_sq = np.dot(v, v)
        new_max_norm_sq = max(max_norm_sq, v_norm_sq)
        
        # Check if v can be added
        can_add = True
        for u in selected:
            dist_sq = np.dot(v - u, v - u)
            if dist_sq < new_max_norm_sq:
                can_add = False
                break
                
        if can_add:
            selected.append(v)
            max_norm_sq = new_max_norm_sq
            
    # Convert to array
    if not selected:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    return np.array(selected, dtype=np.int64)
