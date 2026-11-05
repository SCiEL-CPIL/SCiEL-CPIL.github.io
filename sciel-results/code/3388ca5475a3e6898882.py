import math
import time
import itertools
import numpy as np

def construct():
    n = 5040
    target = 4979
    deadline = time.time() + 50.0  # 50s limit to be safe

    # Generate divisors of 5040
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Filter and sort
    divisors = [d for d in divisors if d <= target]
    divisors.sort()
    N = len(divisors)

    # 1. Try small counts with combinations
    # This is fast for k=1,2,3,4
    # If optimal k is small, this is much faster than DP setup
    for k in range(1, 5):
        if time.time() > deadline:
            break
        # Use combinations
        # We need sum == target
        # Optimization: prune branches if sum > target?
        # itertools.combinations doesn't prune easily.
        # For k=4, 487k combos. Sum check is fast.
        for combo in itertools.combinations(divisors, k):
            if sum(combo) == target:
                return list(combo)
            # Early exit if time is tight?
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # 2. DP for larger counts
    INF = 10**9
    # Use lists for simplicity and to avoid NumPy import issues if any, 
    # but NumPy is available. Lists are fine for 5000 size.
    dp = [INF] * (target + 1)
    dp[0] = 0
    parent = [0] * (target + 1)
    
    # To ensure we don't use a divisor twice, we iterate divisors one by one
    # and update dp backwards? 
    # Standard 0/1 knapsack for min items:
    # for d in divisors:
    #   for s from target down to d:
    #     if dp[s-d] + 1 < dp[s]:
    #        dp[s] = dp[s-d] + 1
    #        parent[s] = d
        
    # This backward iteration avoids the need for a copy of the dp array!
    # This is O(N * T) with low constant factor and no memory copy.
    
    for d in divisors:
        if time.time() > deadline:
            break
        # Iterate backwards from target to d
        # We only need to update if dp[s-d] is not INF
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = d

    if dp[target] >= INF:
        return []

    # Reconstruct
    chosen = []
    s = target
    while s > 0:
        d_val = parent[s]
        if d_val == 0:
            break
        chosen.append(d_val)
        s -= d_val
        
    return chosen
