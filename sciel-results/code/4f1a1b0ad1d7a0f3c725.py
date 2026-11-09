def construct():
    # Step 1: Base-11 construction
    def no_zero_digits_base11(n):
        if n == 0:
            return False
        while n > 0:
            if n % 11 == 0:
                return False
            n //= 11
        return True
    
    S = set()
    for x in range(1, 257):
        if no_zero_digits_base11(x):
            S.add(x)
    
    # Step 2: Local search to improve reciprocal sum
    improved = True
    max_iterations = 500
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        current_sum = sum(1.0/x for x in S)
        
        # Try adding small elements first (highest reciprocal gain)
        for y in range(1, 257):
            if y in S:
                continue
            valid = True
            for a in S:
                s = a + y
                if s % 2 == 0:
                    b = s // 2
                    if b in S:
                        valid = False
                        break
            if valid:
                S.add(y)
                current_sum += 1.0/y
                improved = True
                break
        
        if improved:
            continue
        
        # If no single addition works, try swaps: remove large x, add small y
        for x in sorted(S, reverse=True):
            S_trial = S - {x}
            sum_trial = current_sum - 1.0/x
            
            for y in range(1, x):
                if y in S_trial:
                    continue
                valid = True
                for a in S_trial:
                    s = a + y
                    if s % 2 == 0:
                        b = s // 2
                        if b in S_trial:
                            valid = False
                            break
                if valid:
                    sum_trial += 1.0/y
                    if sum_trial > current_sum:
                        S = S_trial
                        S.add(y)
                        current_sum = sum_trial
                        improved = True
                        break
            
            if improved:
                break
    
    return list(S)
