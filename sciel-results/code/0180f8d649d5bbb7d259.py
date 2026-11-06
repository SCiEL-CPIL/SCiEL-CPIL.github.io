def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()
    
    num_divs = len(divisors)
    
    # Max cardinality is small. 30 is a safe upper bound.
    max_k = 30
    
    # dp[i][k] is a bitset (integer) where bit s is 1 if sum s is achievable
    # using exactly k divisors from the first i divisors.
    dp = [[0] * (max_k + 1) for _ in range(num_divs + 1)]
    dp[0][0] = 1
    
    for i in range(1, num_divs + 1):
        d = divisors[i-1]
        # Start with not using d
        for k in range(max_k + 1):
            dp[i][k] = dp[i-1][k]
        
        # Update using d
        for k in range(1, max_k + 1):
            if dp[i-1][k-1]:
                dp[i][k] |= (dp[i-1][k-1] << d)
                
    # Find minimal k
    min_k = -1
    for k in range(1, max_k + 1):
        if dp[num_divs][k] & (1 << target):
            min_k = k
            break
            
    if min_k == -1:
        raise ValueError("No solution found")
        
    # Backtrack
    result = []
    current_sum = target
    current_k = min_k
    
    for i in range(num_divs, 0, -1):
        d = divisors[i-1]
        # Check if current_sum is achievable with current_k items 
        # using only the first i-1 divisors
        if current_k > 0 and (dp[i-1][current_k] & (1 << current_sum)):
            # d was NOT used
            pass
        else:
            # d WAS used
            result.append(d)
            current_sum -= d
            current_k -= 1
            
    return result
