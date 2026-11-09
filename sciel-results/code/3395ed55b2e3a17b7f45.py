def construct():
    # 1. Generate candidate set based on Base 11, no zero digits
    candidates = []
    for x in range(1, 257):
        num = x
        valid = True
        while num > 0:
            if num % 11 == 0:
                valid = False
                break
            num //= 11
        if valid:
            candidates.append(x)
    
    # 2. Greedy Construction from candidates (ascending order to maximize reciprocal weight)
    S = set()
    for x in candidates:
        # Check if x can be added to S without forming a 3-AP
        can_add = True
        for a in S:
            # Check if (a+x)/2 is in S
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b in S:
                    can_add = False
                    break
            # Check if 2*x - a is in S (i.e., a, x, c is AP)
            c = 2 * x - a
            if c in S:
                can_add = False
                break
        if can_add:
            S.add(x)
            
    # 3. Local Search
    # Try to improve by removing large elements and adding valid small elements from candidates
    improved = True
    # Sort candidates for quick lookup of valid additions
    # We will iterate through S to remove. 
    # To maximize gain, we should try removing larger elements first (lower cost).
    # Or just iterate all.
    while improved:
        improved = False
        current_sum = sum(1.0/x for x in S)
        
        # Sort S in descending order to try removing large elements first
        # Removing a large element is "cheap" in terms of reciprocal loss.
        for x in sorted(list(S), reverse=True):
            S_trial = S - {x}
            sum_trial = current_sum - 1.0/x
            
            added_any = False
            # Try to add any valid y from candidates not in S_trial
            # Sort candidates ascending to prioritize small y (high weight)
            for y in candidates:
                if y in S_trial:
                    continue
                # Check if y is valid with respect to S_trial
                valid = True
                for a in S_trial:
                    s = a + y
                    if s % 2 == 0:
                        b = s // 2
                        if b in S_trial:
                            valid = False
                            break
                    c = 2 * y - a
                    if c in S_trial:
                        valid = False
                        break
                if valid:
                    S_trial.add(y)
                    sum_trial += 1.0/y
                    added_any = True
            
            if sum_trial > current_sum:
                S = S_trial
                current_sum = sum_trial
                improved = True
                # Break to restart the while loop with the new S
                break
                
    return list(S)
