def construct():
    import time
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill
    
    N = 17
    MIN_D = 6
    SIZE = 1 << N
    
    # Precompute weights for all 2^17 words using numpy
    all_words = np.arange(SIZE, dtype=np.int64)
    pop_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.int8)
    
    # Vectorized weight calculation
    w = all_words
    weights = (pop_table[w & 0xFF] + 
               pop_table[(w >> 8) & 0xFF] + 
               pop_table[(w >> 16) & 0xFF])
    
    # Precompute all masks with weight < MIN_D (for forbidden ball marking)
    masks_list = [m for m in range(SIZE) if weights[m] < MIN_D]
    masks = np.array(masks_list, dtype=np.int64)
    
    # Greedy linear code construction
    current_code = np.array([0], dtype=np.int64)
    basis = []
    
    for g in range(1, SIZE):
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
    forbidden = np.zeros(SIZE, dtype=bool)
    for c in current_code:
        forbidden[c ^ masks] = True
    
    added_set = set(current_code.tolist())
    code_size = len(added_set)
    
    # Efficient greedy: find all candidates that are not forbidden, sort by weight,
    # and add them one by one, updating forbidden efficiently
    # Use a priority-queue-like approach with numpy
    
    # Pass 1: decreasing weight (heuristic: heavier words first may block fewer future candidates)
    if time.time() < deadline - 2:
        order1 = np.argsort(-weights)
        for w in order1:
            if time.time() > deadline - 1:
                break
            if not forbidden[w]:
                added_set.add(w)
                forbidden[w ^ masks] = True
                code_size += 1
    
    # Pass 2: increasing weight
    if time.time() < deadline - 2:
        order2 = np.argsort(weights)
        for w in order2:
            if time.time() > deadline - 1:
                break
            if not forbidden[w]:
                added_set.add(w)
                forbidden[w ^ masks] = True
                code_size += 1
    
    # Pass 3: random shuffle with different seed
    if time.time() < deadline - 2:
        rng = np.random.default_rng(42)
        order3 = rng.permutation(SIZE)
        for w in order3:
            if time.time() > deadline - 1:
                break
            if not forbidden[w]:
                added_set.add(w)
                forbidden[w ^ masks] = True
                code_size += 1
    
    # Pass 4: another random shuffle
    if time.time() < deadline - 2:
        rng = np.random.default_rng(123)
        order4 = rng.permutation(SIZE)
        for w in order4:
            if time.time() > deadline - 1:
                break
            if not forbidden[w]:
                added_set.add(w)
                forbidden[w ^ masks] = True
                code_size += 1
    
    # Local search: try to swap a codeword with a non-codeword to increase total count
    # This is a 1-swap local search
    if time.time() < deadline - 3:
        code_list = list(added_set)
        n_code = len(code_list)
        
        # For each codeword, check if removing it and adding a new word helps
        # This is expensive, so we do a limited version
        # Try a few random swaps
        rng = np.random.default_rng(999)
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            # Pick a random codeword to potentially remove
            idx = rng.integers(0, n_code)
            remove_word = code_list[idx]
            
            # Temporarily remove it
            added_set.remove(remove_word)
            forbidden[remove_word ^ masks] = False  # This is wrong - need to recompute
            
            # Actually, recomputing forbidden is expensive. Skip this approach.
            added_set.add(remove_word)
            break  # Skip local search for now due to complexity
    
    result = list(added_set)
    return result
