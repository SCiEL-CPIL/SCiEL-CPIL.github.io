def construct():
    import time
    start_time = time.time()
    
    N = 123457
    # Sieve for largest prime factor
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 1:
            for j in range(i, N + 1, i):
                lpf[j] = i
            
    # Create list of (P(x), x) for x in 1 to N-1
    # P(1) = 1
    items = []
    for x in range(1, N):
        items.append((lpf[x], x))
        
    # Sort by P(x)
    items.sort()
    
    seen = [False] * (N + 1)
    
    i = 0
    n_items = len(items)
    while i < n_items:
        current_max_pf = items[i][0]
        
        # Collect all items with this current_max_pf
        group = []
        while i < n_items and items[i][0] == current_max_pf:
            group.append(items[i][1])
            i += 1
            
        # Check for pairs
        found = False
        # Check against seen (previous groups)
        for x in group:
            y = N - x
            if y > 0 and y < N and seen[y]:
                return [x, y]
            # Check within group
            # To avoid O(G^2), we can use a set for the group
            # But G is likely small for small M.
            # For M=1, G={1}.
            # For M=2, G={1, 2, 4, 8...}.
            # Let's just use a set for the group for O(1) lookup within group.
            
        # If not found against seen, check within group
        if not found:
            group_set = set(group)
            for x in group:
                y = N - x
                if y in group_set:
                    return [x, y]
                    
        # Add group to seen
        for x in group:
            seen[x] = True
            
        # Deadline check
        if time.time() - start_time > 85:
             # Fallback: return best known? 
             # We haven't found one yet. 
             # The seed returns [1, 123456].
             # We should return a valid pair.
             # Since we are iterating from smallest P, if we haven't found one, 
             # it means no pair exists with P <= current_max_pf.
             # But we must return *something*.
             # The loop will eventually find one (at worst P(N-1) or P(1)).
             # Actually, x=1, y=N-1 is always a solution.
             # So we will always find a solution.
             # The deadline is just a safety.
             return [1, N-1]

    # Should not reach here
    return [1, N-1]
