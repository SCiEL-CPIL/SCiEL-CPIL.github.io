def construct():
    import time
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    divisors.sort(reverse=True)
    divisor_set = set(divisors)
    
    # Check size 1
    if target in divisor_set:
        return [target]
        
    # Check size 2
    # d1 + d2 = target
    # Iterate d1
    for i in range(len(divisors)):
        if time.time() > deadline: break
        d1 = divisors[i]
        if d1 >= target: continue
        d2 = target - d1
        if d2 in divisor_set and d2 < d1: # distinct and ordered
            return [d1, d2]
            
    # Check size 3
    # d1 + d2 + d3 = target
    # d1 > d2 > d3
    for i in range(len(divisors)):
        if time.time() > deadline: break
        d1 = divisors[i]
        if d1 >= target: continue
        rem1 = target - d1
        # d2 + d3 = rem1, with d2 < d1
        for j in range(i + 1, len(divisors)):
            if time.time() > deadline: break
            d2 = divisors[j]
            if d2 >= rem1: continue # d3 must be positive
            d3 = rem1 - d2
            if d3 in divisor_set and d3 < d2:
                return [d1, d2, d3]
                
    # Check size 4
    # d1 + d2 + d3 + d4 = target
    # d1 > d2 > d3 > d4
    for i in range(len(divisors)):
        if time.time() > deadline: break
        d1 = divisors[i]
        if d1 >= target: continue
        rem1 = target - d1
        for j in range(i + 1, len(divisors)):
            if time.time() > deadline: break
            d2 = divisors[j]
            if d2 >= rem1: continue
            rem2 = rem1 - d2
            # d3 + d4 = rem2, with d3 < d2
            for k in range(j + 1, len(divisors)):
                if time.time() > deadline: break
                d3 = divisors[k]
                if d3 >= rem2: continue
                d4 = rem2 - d3
                if d4 in divisor_set and d4 < d3:
                    return [d1, d2, d3, d4]
                    
    # Fallback: Greedy (which we know works and gives 5)
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    return chosen
