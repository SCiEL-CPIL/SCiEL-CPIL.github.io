import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Generate all weight 2 vectors (norm squared 2)
    # v[i], v[j] = si, sj
    vecs_w2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vecs_w2.append(v)
    
    vecs_w2 = np.array(vecs_w2, dtype=np.int64)
    k_w2 = len(vecs_w2)
    
    # 2. Generate all weight 4 vectors (norm squared 4)
    # v[i], v[j], v[k], v[l] = si, sj, sk, sl
    vecs_w4_candidates = []
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = [0] * D
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        vecs_w4_candidates.append(v)
                        
    vecs_w4_candidates = np.array(vecs_w4_candidates, dtype=np.int64)
    
    # 3. Greedily select compatible weight 4 vectors
    # Compatibility: dot product <= 2
    selected_w4 = []
    
    # To speed up, we can use a matrix or just list of lists
    # 5280 candidates.
    # Checking against selected list.
    
    # We can pre-compute dot products? 5280x5280 matrix is ~27M entries.
    # 27M ints is ~200MB. Might be too much or slow to create.
    # Let's just do it on the fly.
    
    # Optimization: Sort candidates?
    # Maybe randomize?
    
    for v in vecs_w4_candidates:
        if time.time() > deadline:
            break
            
        compatible = True
        for u in selected_w4:
            dot = np.dot(v, u)
            if dot > 2:
                compatible = False
                break
        if compatible:
            selected_w4.append(v)
            
    selected_w4 = np.array(selected_w4, dtype=np.int64) if selected_w4 else np.empty((0, D), dtype=np.int64)
    
    # Combine
    all_vecs = np.vstack([vecs_w2, selected_w4])
    
    return all_vecs
