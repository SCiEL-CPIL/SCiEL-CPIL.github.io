import time

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors of n
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    
    # Sort divisors, order doesn't strictly matter for correctness of min count, 
    # but might affect which specific set is found.
    # The DP finds the minimum count regardless of order.
    
    # DP setup
    # dp[s] = minimum number of divisors needed to sum to s
    # used[s] = the divisor used to reach s from s-d in the optimal solution found so far
    INF = float('inf')
    dp = [INF] * (target + 1)
    used = [None] * (target + 1)
    
    dp[0] = 0
    
    for d in divs:
        # Iterate downwards to ensure each divisor is used at most once
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                used[s] = d
                
    # Check if solution exists
    if dp[target] == INF:
        return []
        
    # Backtrack to find the divisors
    result = []
    s = target
    while s > 0:
        d = used[s]
        if d is None:
            # Should not happen if dp[target] is not INF
            break
        result.append(d)
        s -= d
        
    return result
