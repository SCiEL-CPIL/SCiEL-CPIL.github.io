import math
import time

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors (not strictly necessary for DP value, but good for consistency)
    divisors.sort()
    
    # DP setup
    # dp[s] = min number of divisors to sum to s
    # Initialize with infinity
    INF = 10**9
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # parent[s] stores the last divisor added to reach sum s
    parent = [None] * (target + 1)
    
    # Fill DP table
    for d in divisors:
        if d > target:
            continue
        # Iterate backwards to ensure 0/1 property
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = d
                
    # Check if solution exists
    if dp[target] == INF:
        # Should not happen for this specific instance
        return []
        
    # Reconstruct the set of divisors
    chosen = []
    s = target
    while s > 0:
        d = parent[s]
        if d is None:
            # Error in reconstruction, break to avoid infinite loop
            break
        chosen.append(d)
        s -= d
        
    return chosen
