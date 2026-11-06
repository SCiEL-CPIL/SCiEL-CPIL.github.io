import math
import time
import numpy as np

def construct():
    start_time = time.time()
    n = 5040
    target = 4979
    
    # Get divisors
    divisors = [i for i in range(1, n + 1) if n % i == 0]
    
    INF = float('inf')
    # dp[s] = min number of divisors to sum to s
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # prev_div[s] stores the divisor d that was used to reach s optimally
    # prev_sum[s] stores the previous sum s-d
    prev_div = [0] * (target + 1)
    prev_sum = [0] * (target + 1)
    
    # Sort divisors in descending order for better pruning potential
    divisors.sort(reverse=True)
    
    # Process each divisor
    for d in divisors:
        # Iterate backwards to ensure 0/1 property (each divisor used at most once)
        # s must be at least d
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                prev_div[s] = d
                prev_sum[s] = s - d
                
        # Time check
        if time.time() - start_time > 50:
            break

    if dp[target] == INF:
        # No solution found
        return []

    # Reconstruct
    result = []
    s = target
    while s > 0:
        d = prev_div[s]
        if d == 0:
            # Safety break
            break
        result.append(d)
        s = prev_sum[s]
        
    if time.time() - start_time > 55:
         pass

    return sorted(result, reverse=True)
