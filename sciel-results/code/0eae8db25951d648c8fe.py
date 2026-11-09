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
    # We need masks for distance < 6, i.e., weight 0 to 5
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
    
    # Initialize base forbidden mask from the linear code
    base_forbidden = np.zeros(1 << N, dtype=bool)
    for c in current_code:
        base_forbidden[c ^ masks] = True
        
    best_code = set(current_code.tolist())
    best_size = len(best_code)
    
    # Function to run a greedy pass given an initial forbidden state
    def run_greedy_pass(forbidden_init, order):
        forbidden = forbidden_init.copy()
        added = 0
        for w in order:
            if time.time() > deadline - 1:
                break
            if not forbidden[w]:
                added += 1
                forbidden[w ^ masks] = True
        return added, forbidden

    # We will run multiple passes with different orderings
    # Pass 1: Decreasing weight
    order1 = np.argsort(-weights)
    # Pass 2: Increasing weight
    order2 = np.argsort(weights)
    
    # Random orders
    rng = np.random.default_rng(42)
    
    # List of orders to try
    orders = [order1, order2]
    
    # Add random orders
    while time.time() < deadline - 5:
        orders.append(rng.permutation(1 << N))
        if len(orders) > 20: # Cap the number of pre-generated orders to save memory/time
            break
            
    for order in orders:
        if time.time() > deadline - 2:
            break
            
        added, final_forbidden = run_greedy_pass(base_forbidden, order)
        # The number of words is len(current_code) + added
        # But we need the actual words to return them.
        # The function above only counts. We need to track the words.
        
        # Let's rewrite the pass to track words
        pass

    # Revised strategy: Track words in the pass
    def run_greedy_pass_with_words(forbidden_init, order):
        forbidden = forbidden_init.copy()
        new_words = []
        for w in order:
            if time.time() > deadline - 1:
                break
            if not forbidden[w]:
                new_words.append(w)
                forbidden[w ^ masks] = True
        return new_words

    # Try the orders
    for order in orders:
        if time.time() > deadline - 2:
            break
        new_words = run_greedy_pass_with_words(base_forbidden, order)
        current_set = set(current_code.tolist())
        current_set.update(new_words)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_code = current_set
            
    return list(best_code)
