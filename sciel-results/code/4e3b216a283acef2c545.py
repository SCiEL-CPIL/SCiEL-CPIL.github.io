"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Generate candidate vectors with norm 2
    # Types:
    # A: +/- 2e_i (22 vectors)
    # B: +/- e_i +/- e_j +/- e_k +/- e_l (5280 vectors)
    
    candidates = []
    
    # Type A
    for i in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[i] = sign * 2
            candidates.append(v)
            
    # Type B
    for indices in itertools.combinations(range(D), 4):
        i, j, k, l = indices
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    for s4 in (1, -1):
                        v = [0] * D
                        v[i] = s1
                        v[j] = s2
                        v[k] = s3
                        v[l] = s4
                        candidates.append(v)
                        
    candidates = np.array(candidates, dtype=np.int64)
    n_cands = len(candidates)
    
    # 2. Greedy selection
    # We need a set S such that for all u, v in S, ||u-v|| >= 2.
    # Since all candidates have norm 2, max norm is 2.
    # So we just need min pairwise distance >= 2.
    
    # Optimization: 
    # Two vectors have dist < 2 iff dist^2 < 4.
    # dist^2 is integer. So dist^2 can be 1 or 2.
    # dist^2 = 1: differ in 1 coord by 1.
    # dist^2 = 2: differ in 2 coords by 1.
    
    # To speed up, we can use a simple greedy loop.
    # With 5302 candidates, O(N^2) is ~28M ops, which might be slow in Python.
    # We need a faster conflict check or a smarter selection.
    
    # Heuristic: Sort by "isolation" or just use a spatial index?
    # Given the low dimension (11) and small values (-2 to 2), we can use a dictionary for lookups?
    # No, we need to check against ALL selected points.
    
    # Let's try a random-restart greedy with a few iterations if time permits.
    # Or just one greedy pass with a good ordering.
    # Ordering: Random shuffle might work.
    
    import random
    random.seed(42)
    
    selected = []
    selected_set = set() # Store tuples for O(1) lookup? No, we need distance check.
    
    # To make it faster, we can pre-compute the "conflict" neighborhoods?
    # No, let's just do a nested loop but break early if we exceed deadline.
    # 5302 * 5302 / 2 is 14M pairs. In Python, this is ~5-10 seconds.
    # So we can do a full greedy pass.
    
    indices = list(range(n_cands))
    random.shuffle(indices)
    
    for idx in indices:
        if time.time() > deadline:
            break
        v = candidates[idx]
        is_valid = True
        # Check against all already selected
        for u in selected:
            diff = v - u
            dist_sq = np.dot(diff, diff)
            if dist_sq < 4:
                is_valid = False
                break
        
        if is_valid:
            selected.append(v)
            
    if not selected:
        # Fallback to seed if something went wrong
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    return np.array(selected, dtype=np.int64)

# EVOLVE-BLOCK-END
