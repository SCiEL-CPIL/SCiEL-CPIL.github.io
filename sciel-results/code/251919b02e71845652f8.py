import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    # Generate all integer vectors with squared norm 4
    candidates = []
    
    # Type 1: one coordinate ±2 (squared norm 4)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            candidates.append(tuple(v))
    
    # Type 2: two coordinates ±1 (squared norm 2) - wait, this is norm sqrt(2), not 2
    # I need squared norm 4. Let me reconsider.
    # Squared norm 4: 
    #   - One ±2: 22 vectors
    #   - Four ±1: C(11,4)*16 = 5280 vectors
    #   - Two ±1 and... no, two ±1 gives squared norm 2.
    #   - One ±2 and... no.
    #   - Actually: sum of squares = 4. Options: 4+0+... or 1+1+1+1+...
    #   So: one ±2, or four ±1's.
    
    # Let me redo:
    candidates = []
    
    # One ±2
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
    
    # Four ±1's
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cands = len(candidates)
    
    # Greedy: all have squared norm 4, so max_sq_norm = 4
    # Need ||u-v||^2 >= 4 for all pairs, i.e., u·v <= 2
    selected = []
    
    for v in candidates:
        if time.time() > deadline:
            break
        
        if not selected:
            selected.append(v)
            continue
        
        sel = np.array(selected, dtype=np.int64)
        dots = sel @ v
        if np.all(dots <= 2):
            selected.append(v)
    
    if not selected:
        # Fallback
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return np.array(selected, dtype=np.int64)
