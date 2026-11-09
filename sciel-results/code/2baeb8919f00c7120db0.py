def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    best_seq = []
    best_len = 0
    
    # Strategy 1: Standard Ascending Greedy
    seq = try_greedy(range(2, 129))
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = seq

    # Strategy 2: Randomized Greedy
    # We will run many iterations until deadline
    candidates = list(range(2, 129))
    
    while time.time() < deadline:
        random.shuffle(candidates)
        
        seq = try_greedy(candidates)
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
            
            # Early exit if we hit a very high bound? 
            # Max possible length is 127. 
            # If we get > 100, it's very good.
            if best_len > 100:
                break
        
        # Check time
        if time.time() > deadline - 1.0:
            break

    return best_seq
