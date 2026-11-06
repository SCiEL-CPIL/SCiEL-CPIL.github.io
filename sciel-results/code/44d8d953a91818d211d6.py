def construct():
    N = 512
    
    # Find M in [1, 512] with the maximum number of divisors
    # Any set of divisors of M is a valid clique because lcm(a,b) divides M.
    # We want to maximize the number of divisors.
    max_divs = 0
    best_M = 1
    
    for M in range(1, N + 1):
        count = 0
        # Count divisors of M
        i = 1
        while i * i <= M:
            if M % i == 0:
                count += 1
                if i != M // i:
                    count += 1
            i += 1
        
        if count > max_divs:
            max_divs = count
            best_M = M
            
    # Return the divisors of best_M
    # This set has cardinality max_divs (which is 24 for N=512)
    # and satisfies the condition lcm(a,b) <= 512 for all pairs.
    result = [d for d in range(1, N + 1) if best_M % d == 0]
    
    return result
