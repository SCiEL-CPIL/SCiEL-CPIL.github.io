<<<<<<< SEARCH
def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    # Run multiple restarts
    while time.time() < deadline:
        # Check if we have enough time for a full greedy + local search
        if time.time() + 10.0 > deadline:
            break
        
        # Alternate between greedy and hypergraph peeling strategies
        if seed % 2 == 0:
            blocks = _greedy_random(v, k, t, seed)
        else:
            blocks = _hypergraph_peel(v, k, t, seed, deadline)
        
        # Apply local search if we have time
        if time.time() + 5.0 < deadline:
            rng = random.Random(seed + 1000000)
            blocks = _local_search(v, k, t, blocks, rng, deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
        # Stop early if we hit theoretical lower bound
        if best_len == 69:
            break
            
        seed += 1
    
    return best_blocks
=======
def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin before 150s kill
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    # Run multiple restarts
    while time.time() < deadline:
        # Check if we have enough time for a full greedy + local search
        remaining = deadline - time.time()
        if remaining < 8.0:
            break
        
        # Alternate between greedy and hypergraph peeling strategies
        if seed % 2 == 0:
            blocks = _greedy_random(v, k, t, seed)
        else:
            blocks = _hypergraph_peel(v, k, t, seed, deadline)
        
        # Apply local search if we have time
        if time.time() + 3.0 < deadline:
            rng = random.Random(seed + 1000000)
            blocks = _local_search(v, k, t, blocks, rng, deadline)
        
        if blocks is not None and len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
        # Stop early if we hit theoretical lower bound
        if best_len == 69:
            break
            
        seed += 1
    
    return best_blocks
>>>>>>> REPLACE
