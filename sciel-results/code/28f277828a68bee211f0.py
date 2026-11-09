def construct():
    import time
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    N = 17
    MIN_D = 6
    
    # Greedy construction of a linear code
    # We try to find a basis of size k such that all combinations have weight >= MIN_D
    # This gives a code of size 2^k
    
    basis = []
    
    # Precompute weights for all possible words? 
    # 2^17 = 131072. 
    # We can just compute on the fly.
    
    # To speed up, we can iterate candidates in a specific order.
    # Or just iterate 0 to 2^N - 1.
    
    # We want to find up to 8 rows (size 256)
    # If we find 8, we stop.
    
    # We can try to find rows in a greedy manner.
    # For each step k, we scan all candidates and add the first one that works.
    # This might be slow if we scan 131k candidates for each of 8 steps.
    # 8 * 131k = 1M iterations.
    # Each iteration involves checking against 2^(k-1) existing combinations.
    # Total checks: sum_{k=1}^8 2^(k-1) = 255.
    # So 1M * 255 = 255M operations.
    # This might be too slow in Python.
    
    # Optimization: 
    # We can maintain a list of all valid combinations of the current basis.
    # When adding a new candidate g, we check if wt(g + c) >= MIN_D for all c in current_code.
    # If we find a g that works, we add it to the basis and update current_code.
    
    # To speed up the search for g, we can randomize the order or use a heuristic.
    # But a simple scan might work if we break early.
    
    # Let's try a randomized greedy approach with restarts.
    # Or, better, let's try to find the specific 256-word code.
    
    # Alternative: Use the known 256-word code structure if possible.
    # But let's try the greedy linear code construction first.
    
    # We can speed up the weight check by precomputing weights of all 2^17 words?
    # 131k integers. Memory is fine.
    # Time to precompute: 131k * bit_count. Fast.
    
    weights = [0] * (1 << N)
    for i in range(1, 1 << N):
        weights[i] = weights[i >> 1] + (i & 1)
        
    # Now weight check is O(1).
    
    # Greedy basis construction
    basis = []
    current_code = [0]  # List of all combinations of current basis
    
    # We want to find 8 rows.
    # We scan candidates in a fixed order.
    # To avoid bias, we can shuffle? No, deterministic is better for reproducibility.
    # But the order matters.
    # Let's try to add rows that are "far" from existing ones.
    
    # We'll iterate through candidates.
    # For each candidate g, check if it can be added.
    # If yes, add it and update current_code.
    # We stop when basis size is 8 or time is up.
    
    # To make it faster, we can check candidates in an order that prioritizes high weight.
    # But any g with wt >= 6 is fine for the first row.
    
    # Let's just scan 0 to 2^N - 1.
    # But we must ensure we don't pick g=0.
    
    for g in range(1, 1 << N):
        if time.time() > deadline:
            break
            
        # Check if g can be added to the basis
        # Condition: wt(g + c) >= MIN_D for all c in current_code
        valid = True
        for c in current_code:
            if weights[g ^ c] < MIN_D:
                valid = False
                break
        
        if valid:
            basis.append(g)
            # Update current_code
            # New code = old code union (old code + g)
            # Since g is linearly independent (wt(g)>=6 implies g!=0), and we check distance, 
            # g is not in the span? 
            # Actually, if g is in the span, then g+c=0 for some c, wt=0 < 6. 
            # So g is linearly independent.
            new_code = []
            for c in current_code:
                new_code.append(c)
                new_code.append(c ^ g)
            current_code = new_code
            
            if len(basis) >= 8:
                break
                
    # If we found 8 rows, we have 256 words.
    # If we found fewer, we have 2^k words.
    # We can also try to add non-linear words if we have time.
    # But the linear code is a good baseline.
    
    # To improve, we can try to find more rows if we didn't reach 8.
    # But the loop above already tries to add as many as possible.
    
    # If we have time, we can try to add non-linear words to the linear code.
    # But the linear code of size 256 is already very good.
    
    # Let's return the current_code
    # But we must ensure distinct integers.
    # current_code contains distinct integers.
    
    return current_code
