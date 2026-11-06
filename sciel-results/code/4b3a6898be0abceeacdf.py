<<<<<<< SEARCH
    best = set()
    best_size = 0
    
    # Strategy 1: Min-conflict greedy (deterministic)
    S = min_conflict_greedy()
    S = local_search(S)
    if len(S) > best_size:
        best_size = len(S)
        best = set(S)
=======
    best = set()
    best_size = 0
    
    # Strategy 1: Valid greedy (deterministic)
    S = valid_greedy()
    S = local_search(S)
    if len(S) > best_size and validate(S):
        best_size = len(S)
        best = set(S)
>>>>>>> REPLACE

