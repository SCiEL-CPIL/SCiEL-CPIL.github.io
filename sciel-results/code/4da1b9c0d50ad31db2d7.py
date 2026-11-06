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
    
    # Sort divisors descending
    divs.sort(reverse=True)
    
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # We want to minimize the number of divisors.
    # Let's try k = 1, 2, 3, ... and check if a subset of size k sums to target.
    # Since the number of divisors is small (60), and target is 4979, 
    # we can use a DP that tracks the minimum number of items to reach a sum.
    
    # dp[s] = min number of divisors needed to sum to s
    # We initialize with infinity.
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # used[s] = divisor d such that dp[s] = dp[s-d] + 1
    # We need to be careful with 0/1 knapsack. 
    # The standard 0/1 knapsack for min items:
    # Iterate through each divisor d.
    # For s from target down to d:
    #   if dp[s-d] + 1 < dp[s]:
    #      dp[s] = dp[s-d] + 1
    #      used[s] = d
    # This works because we iterate s downwards, ensuring each divisor is used at most once.
    
    used = [None] * (target + 1)
    
    for d in divs:
        # If d > target, we can't use it for any sum <= target
        if d > target:
            continue
            
        # Iterate backwards to ensure 0/1 property
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                used[s] = d
                
        # Time check
        if time.time() > deadline:
            break
            
    if dp[target] == INF:
        # Fallback: Greedy approach (largest divisors first)
        # This should not happen if a solution exists, but good for safety.
        remainder = target
        chosen = []
        for d in divs:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        return chosen
        
    # Backtrack to find the divisors
    result = []
    s = target
    while s > 0:
        d = used[s]
        if d is None:
            # This should not happen if dp[target] != INF
            break
        result.append(d)
        s -= d
        
    return result
