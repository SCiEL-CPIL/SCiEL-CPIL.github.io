def construct():
    import time
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill
    
    N = 17
    MIN_D = 6
    
    # Precompute weights for all 2^17 words using numpy
    all_words = np.arange(1 << N, dtype=np.int64)
    pop_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.int8)
    
    # Vectorized weight calculation
    w = all_words
    weights = (pop_table[w & 0xFF] + 
               pop_table[(w >> 8) & 0xFF] + 
               pop_table[(w >> 16) & 0xFF])
    
    # Precompute all masks with weight < MIN_D (for forbidden ball marking)
    masks_list = [m for m in range(1 << N) if weights[m] < MIN_D]
    masks = np.array(masks_list, dtype=np.int64)
    
    # Greedy linear code construction
    current_code = np.array([0], dtype=np.int64)
    basis = []
    
    for g in range(1, 1 << N):
        if time.time() > deadline:
            break
        
        xor_vals = g ^ current_code
        min_dist = np.min(weights[xor_vals])
        
        if min_dist >= MIN_D:
            basis.append(g)
            new_code = np.concatenate([current_code, current_code ^ g])
            current_code = new_code
            
            if len(basis) >= 8:
                break
    
    # Initialize forbidden mask
    forbidden = np.zeros(1 << N, dtype=bool)
    for c in current_code:
        forbidden[c ^ masks] = True
    
    added_set = set(current_code.tolist())
    
    # Multiple greedy passes with different orderings to find more non-linear words
    # Pass 1: decreasing weight
    order1 = np.argsort(-weights)
    for w in order1:
        if time.time() > deadline:
            break
        if not forbidden[w]:
            added_set.add(w)
            forbidden[w ^ masks] = True
    
    # Pass 2: increasing weight (different heuristic may find different words)
    order2 = np.argsort(weights)
    for w in order2:
        if time.time() > deadline:
            break
        if not forbidden[w]:
            added_set.add(w)
            forbidden[w ^ masks] = True
    
    # Pass 3: random shuffle
    if time.time() < deadline - 5:
        rng = np.random.default_rng(42)
        order3 = rng.permutation(1 << N)
        for w in order3:
            if time.time() > deadline:
                break
            if not forbidden[w]:
                added_set.add(w)
                forbidden[w ^ masks] = True
    
    # Pass 4: another random shuffle with different seed
    if time.time() < deadline - 3:
        rng = np.random.default_rng(123)
        order4 = rng.permutation(1 << N)
        for w in order4:
            if time.time() > deadline:
                break
            if not forbidden[w]:
                added_set.add(w)
                forbidden[w ^ masks] = True
    
    result = list(added_set)
    return result
