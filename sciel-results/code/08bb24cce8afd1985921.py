"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    vectors = []
    
    # 1. Add all Weight 2, Value +/-2 vectors (Norm^2 = 8)
    # These are mutually compatible.
    for pos in itertools.combinations(range(D), 2):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = np.zeros(D, dtype=np.int64)
                v[pos[0]] = 2 * s1
                v[pos[1]] = 2 * s2
                vectors.append(v)
                
    vec_array = np.array(vectors)
    
    # 2. Add Weight 2, Value +/-1 vectors (Norm^2 = 2)
    # Must be compatible with existing (always true, max dot 4 <= 4)
    # Must be compatible with each other: dot <= -2
    # We will greedily add them.
    w2v1_candidates = []
    for pos in itertools.combinations(range(D), 2):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = np.zeros(D, dtype=np.int64)
                v[pos[0]] = s1
                v[pos[1]] = s2
                w2v1_candidates.append(v)
                
    w2v1_array = np.array(w2v1_candidates)
    n_w2v1 = len(w2v1_array)
    
    # Greedy selection for W2V1
    # Sort by norm? All same.
    # Check compatibility with current vec_array and among themselves.
    # Dot with vec_array (W2V2) is always <= 4, so OK.
    # Dot with other W2V1 must be <= -2.
    
    in_set_w2v1 = np.zeros(n_w2v1, dtype=bool)
    # Precompute dot products among W2V1
    dots_w2v1 = w2v1_array @ w2v1_array.T
    
    # Greedy: add if dot with all existing in_set_w2v1 is <= -2
    # To speed up, we can just iterate.
    # Since n_w2v1 is 220, 220*220 is small.
    
    # We can actually just pick one from each support pair?
    # Supports are 55. For each support, we have 4 vectors.
    # v1=(1,1), v2=(1,-1), v3=(-1,1), v4=(-1,-1).
    # v1 dot v4 = -2. OK.
    # v2 dot v3 = -2. OK.
    # v1 dot v2 = 0. Conflict.
    # So we can pick {v1, v4} or {v2, v3} or mix?
    # v1 and v3: dot 0. Conflict.
    # So max 2 per support.
    # We will just greedily add.
    
    for i in range(n_w2v1):
        # Check against already added
        if not in_set_w2v1[i]:
             # Check if compatible with all j where in_set_w2v1[j] is True
             # dots_w2v1[i, in_set_w2v1] <= -2
             if np.all(dots_w2v1[i, in_set_w2v1] <= -2):
                 in_set_w2v1[i] = True
                 
    selected_w2v1 = w2v1_array[in_set_w2v1]
    if len(selected_w2v1) > 0:
        vec_array = np.vstack([vec_array, selected_w2v1])
        
    # 3. Add Weight 8, Value +/-1 vectors (Norm^2 = 8)
    # Must be compatible with existing (W2V2 and W2V1).
    # W8V1 dot W2V2 <= 4 (Always OK).
    # W8V1 dot W2V1 <= 4 (Always OK, max 2*1+... wait.
    # W8V1 has 8 ones. W2V1 has 2 ones.
    # Max overlap 2. Dot max 2. OK.
    
    # Must be compatible with each other: dot <= 4.
    
    w8v1_candidates = []
    for pos in itertools.combinations(range(D), 8):
        for signs in itertools.product([1, -1], repeat=8):
            v = np.zeros(D, dtype=np.int64)
            for p, s in zip(pos, signs):
                v[p] = s
            w8v1_candidates.append(v)
            
    w8v1_array = np.array(w8v1_candidates)
    n_w8v1 = len(w8v1_array) # 42240
    
    # Greedy selection for W8V1
    # This is larger. 42240.
    # We can't compute full 42240x42240 dot matrix easily in memory if not careful?
    # 42240^2 ~ 1.8e9. Too big.
    # We need online checking.
    
    # Sort candidates? Randomize?
    # Greedy: Iterate, add if compatible with all already added.
    # To speed up, we can maintain a list of added vectors.
    # Checking against all added is O(k). Total O(k^2).
    # If k is ~1000, 1e6 ops. Fine.
    # If k is ~10000, 1e8 ops. Might be slow in Python.
    # We can use numpy for batch checks if we add in batches, but greedy is sequential.
    
    # Optimization:
    # We can check compatibility with the 220 W2V2 and ~110 W2V1 vectors first (fast).
    # Then check against selected W8V1.
    
    base_vectors = vec_array
    n_base = len(base_vectors)
    
    selected_w8v1 = []
    
    # To speed up, we can pre-check against base_vectors
    # But we do it on the fly.
    
    # Time limit check
    # We have ~100s left.
    
    # We can try a smarter greedy:
    # Group by support?
    # Supports of size 8.
    
    # Let's just do a simple greedy with a time break.
    # Shuffle to avoid bias?
    rng = np.random.default_rng(42)
    order = rng.permutation(n_w8v1)
    
    for idx in order:
        if time.time() > deadline - 5:
            break
            
        v = w8v1_array[idx]
        
        # Check against base_vectors
        # base_vectors includes W2V2 and W2V1.
        # Max dot with W2V2 is 4. OK.
        # Max dot with W2V1 is 2. OK.
        # So we only need to check against selected_w8v1.
        
        if len(selected_w8v1) == 0:
            selected_w8v1.append(v)
            continue
            
        # Check against selected_w8v1
        # Convert selected_w8v1 to array for vectorized dot?
        # This is inefficient if done every step.
        # Better to keep it as a list and convert periodically?
        # Or just loop.
        
        # For speed, we can maintain a numpy array of selected W8V1.
        # But appending to numpy array is slow.
        
        # Let's use a list and check.
        # If len is small, it's fast.
        
        compatible = True
        # We can vectorize the check if we keep selected_w8v1 in a numpy array
        # but resizing is bad.
        
        # Alternative:
        # Keep selected_w8v1 in a list.
        # Check dot products.
        for w in selected_w8v1:
            if np.dot(v, w) > 4:
                compatible = False
                break
                
        if compatible:
            selected_w8v1.append(v)
            
    if len(selected_w8v1) > 0:
        vec_array = np.vstack([vec_array, np.array(selected_w8v1)])
        
    return vec_array

# EVOLVE-BLOCK-END
