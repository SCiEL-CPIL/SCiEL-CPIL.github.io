    num_restarts = 1000
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        S = set(numbers)
        
        # ... greedy removal ...
        
        # Local search
        S = local_search(S, max_iters=500)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
