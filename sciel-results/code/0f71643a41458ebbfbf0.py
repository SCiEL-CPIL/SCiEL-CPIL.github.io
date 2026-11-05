<<<<<<< SEARCH
    # 1. Greedy Initialization
    uncovered = all_mask
    code = []
    
    # Sort words by some heuristic? Or just iterate.
    # To make greedy deterministic and fast, we can just pick the best one each time.
    # Optimization: Precompute "potential" coverage?
    # Simple greedy:
    while uncovered:
        best_i = -1
        best_cov = 0
        # Iterate all words
        for i in range(N):
            cov = (balls[i] & uncovered).bit_count()
            if cov > best_cov:
                best_cov = cov
                best_i = i
        code.append(best_i)
        uncovered &= ~balls[best_i]
=======
    # 1. Greedy Initialization with randomization and restart
    import random
    best_code = None
    best_len = float('inf')
    
    for attempt in range(5):
        random.seed(42 + attempt)
        uncovered = all_mask
        code = []
        
        # Randomized greedy: pick from top-k candidates
        k = 3 if attempt < 3 else 5
        while uncovered:
            # Find top-k candidates
            scores = [( (balls[i] & uncovered).bit_count(), i ) for i in range(N)]
            scores.sort(reverse=True)
            top_k = scores[:k]
            
            # Pick randomly from top-k
            chosen = random.choice(top_k)
            best_i = chosen[1]
            code.append(best_i)
            uncovered &= ~balls[best_i]
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = code[:]
    
    code = best_code
    uncovered = all_mask
    for i in code:
        uncovered &= ~balls[i]
>>>>>>> REPLACE
