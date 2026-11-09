import numpy as np
import itertools
import time

D = 11
DEADLINE = 110.0

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time

    # Part 1: All 220 vectors of Type 1
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
    
    # Part 2: Type 2 candidates (Even parity)
    # 165 supports, 128 sign patterns each
    candidates = []
    for support in itertools.combinations(range(D), 8):
        # Generate 128 even parity sign patterns
        # Iterate over first 7 signs, determine 8th
        for signs_7 in itertools.product([-1, 1], repeat=7):
            # Product of first 7
            prod_7 = 1
            for s in signs_7:
                prod_7 *= s
            # 8th sign must be prod_7 to make total product 1
            sign_8 = prod_7
            signs = list(signs_7) + [sign_8]
            
            v = [0] * D
            for k, idx in enumerate(support):
                v[idx] = signs[k]
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int8)
    n_cand = len(candidates)
    
    # Compute conflict matrix
    # C[i, j] = 1 if candidates[i] . candidates[j] > 4
    # Use float for dot product to avoid overflow issues? int8 dot int8 might overflow?
    # Max dot product is 8. int8 range is -128 to 127. Safe.
    # But numpy dot on int8 returns int8? Or int32?
    # Let's use int16 or int32 to be safe.
    candidates_int = candidates.astype(np.int16)
    
    # Matrix multiply
    # This might take a few seconds
    dots = candidates_int @ candidates_int.T
    
    # Conflict if dot > 4
    conflict = dots > 4
    
    # Greedy Max Independent Set
    # We want to select a subset of indices such that no two selected are connected in 'conflict'
    # Greedy:
    # 1. Shuffle indices
    # 2. Iterate
    # 3. If index is still "available", select it, and mark it and its neighbors as "unavailable"
    
    rng = np.random.default_rng(42)
    order = rng.permutation(n_cand)
    
    selected = []
    available = np.ones(n_cand, dtype=bool)
    
    for idx in order:
        if not available[idx]:
            continue
            
        # Select idx
        selected.append(idx)
        
        # Mark idx and its neighbors as unavailable
        # neighbors are where conflict[idx] is True
        available[idx] = False
        available[conflict[idx]] = False
        
        if elapsed() > DEADLINE:
            break
            
    # Build final list
    final_vectors = vectors.copy()
    if selected:
        selected_vecs = candidates[selected]
        final_vectors.extend(selected_vecs.tolist())
        
    return np.array(final_vectors, dtype=np.int64)
