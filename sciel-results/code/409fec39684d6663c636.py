<<<<<<< SEARCH
    # Run multiple passes with different seeds to explore the solution space
    # Each pass is very fast, so we can run many.
    num_passes = 1000 # Adjust based on time
    
    for i in range(num_passes):
        elapsed = time.time() - start
        if elapsed > deadline - 2: # 2s safety margin
            break
            
        # Use a deterministic seed for reproducibility in case of timeout, 
        # but vary it to explore different paths.
        # We can also mix deterministic and random strategies.
        seed = i
        result = _greedy_pass(primes, seed, deadline - (time.time() - start))
        
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
=======
    # Run multiple passes with different seeds to explore the solution space
    # Each pass is very fast, so we can run many until the deadline.
    i = 0
    while True:
        elapsed = time.time() - start
        if elapsed > deadline - 2: # 2s safety margin
            break
            
        # Use a deterministic seed for reproducibility in case of timeout, 
        # but vary it to explore different paths.
        seed = i
        result = _greedy_pass(primes, seed, deadline - (time.time() - start))
        
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
        i += 1
>>>>>>> REPLACE
