import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Generate all weight-3 vectors
    vectors = []
    for support in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, val in zip(support, signs):
                v[idx] = val
            vectors.append(np.array(v, dtype=np.int64))
            
    # Greedy selection
    selected = []
    for v in vectors:
        if time.time() > deadline:
            break
        valid = True
        for u in selected:
            dist_sq = np.sum((v - u) ** 2)
            if dist_sq < 3:  # Norm squared of weight 3 is 3
                valid = False
                break
        if valid:
            selected.append(v)
            
    if not selected:
        # Fallback to seed if something goes wrong
        vectors_seed = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors_seed.append(v)
        return np.array(vectors_seed, dtype=np.int64)
        
    return np.array(selected, dtype=np.int64)
