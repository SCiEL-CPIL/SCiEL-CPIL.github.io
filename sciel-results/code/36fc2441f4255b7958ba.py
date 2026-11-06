def construct():
    from math import gcd
    import time
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    N = 511
    weights = [1.0 / (N + 1 - a) for a in range(1, N + 1)]
    
    # Greedy initialization
    chosen = []
    for a in range(N, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
    
    def score(sol):
        return sum(weights[a-1] for a in sol)
    
    current_score = score(chosen)
    
    # Local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element
        for i, a in enumerate(list(chosen)):
            if time.time() > deadline:
                break
            # Remove a
            new_chosen = chosen[:i] + chosen[i+1:]
            # Find candidates that were blocked by a
            # Candidates are numbers not in new_chosen
            # that are coprime to all in new_chosen
            # We can just iterate all numbers not in new_chosen
            # and check if they are coprime to new_chosen
            # To be efficient, we can just check numbers that share a factor with a
            # because only those were blocked by a
            # But some numbers might be blocked by others too.
            # So we just check all numbers not in new_chosen.
            
            # Optimization: Only check numbers that are coprime to a? No, they are blocked by a.
            # We want to add numbers that are coprime to new_chosen.
            # Let's just iterate all a' not in new_chosen.
            
            added_any = False
            # Sort candidates by weight descending
            # Candidates: all a' in 1..N not in new_chosen
            # We can pre-compute list of all numbers
            # and filter
            
            # For speed, just iterate 1..N
            for b in range(N, 0, -1):
                if b in new_chosen:
                    continue
                # Check if b is coprime to all in new_chosen
                if all(gcd(b, c) == 1 for c in new_chosen):
                    new_chosen.append(b)
                    added_any = True
                    # We can stop adding if we find enough?
                    # No, we want the best set.
                    # But adding more might block better ones?
                    # Greedy on the remaining is safe.
            
            if added_any:
                new_score = score(new_chosen)
                if new_score > current_score:
                    chosen = new_chosen
                    current_score = new_score
                    improved = True
    
    return chosen
