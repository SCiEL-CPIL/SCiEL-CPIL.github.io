    # 1. Greedy to find a good start
    greedy_set = greedy_extend(current)
    if len(greedy_set) > best_size:
        best_size = len(greedy_set)
        best_set = greedy_set
        
    # 2. Backtracking to try to beat it
    # We should start backtracking from the prescribed set again, but now with a high best_size.
    # To make backtracking faster, we can also use the greedy set as a hint? 
    # No, just use the high best_size for pruning.
    
    backtrack(current, 12)
