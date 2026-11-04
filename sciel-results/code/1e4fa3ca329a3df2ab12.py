import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Generate candidate vectors
    candidates = []
    
    # Norm sq 2: Two non-zero entries +-1
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.append((2, v)) # (norm_sq, vector)

    # Norm sq 4: 
    # Type 1: One non-zero entry +-2
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            candidates.append((4, v))
            
    # Type 2: Four non-zero entries +-1
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in zip(idxs, signs):
                v[k] = s
            candidates.append((4, v))

    # Norm sq 8:
    # Type 1: Two non-zero entries +-2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = 2*si, 2*sj
                candidates.append((8, v))
                
    # Type 2: One +-2, Four +-1
    for idx2 in range(D):
        for s2 in (1, -1):
            for idx1s in itertools.combinations([i for i in range(D) if i != idx2], 4):
                for signs1 in itertools.product([1, -1], repeat=4):
                    v = [0] * D
                    v[idx2] = 2 * s2
                    for k, s in zip(idx1s, signs1):
                        v[k] = s
                    candidates.append((8, v))

    # Sort by norm squared, then by index for determinism
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    selected = []
    max_norm_sq = 0
    
    for norm_sq, v in candidates:
        if time.time() > deadline:
            break
            
        # Check if adding v violates constraints
        # New max norm sq will be max(max_norm_sq, norm_sq)
        new_max_norm_sq = max(max_norm_sq, norm_sq)
        
        valid = True
        for u in selected:
            # Calculate distance squared
            dist_sq = sum((a - b) ** 2 for a, b in zip(v, u))
            if dist_sq < new_max_norm_sq:
                valid = False
                break
                
        if valid:
            selected.append(v)
            max_norm_sq = new_max_norm_sq

    if not selected:
        # Fallback to simple set if something went wrong
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

    return np.array(selected, dtype=np.int64)
