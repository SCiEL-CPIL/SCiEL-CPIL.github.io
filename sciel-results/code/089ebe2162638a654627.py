def construct():
    # Generate divisors of 5040
    n = 5040
    target = 4979
    divisors = [d for d in range(1, n+1) if n % d == 0]
    divisors.sort()
    
    # dp[s] = minimum number of divisors to achieve sum s
    # Initialize with infinity
    INF = float('inf')
    max_sum = sum(divisors)
    dp = [INF] * (max_sum + 1)
    dp[0] = 0
    
    # parent[s] = (prev_sum, divisor_used) for reconstruction
    parent = [None] * (max_sum + 1)
    
    # 0/1 knapsack: for each divisor, update dp in reverse
    for d in divisors:
        for s in range(max_sum - d, -1, -1):
            if dp[s] + 1 < dp[s + d]:
                dp[s + d] = dp[s] + 1
                parent[s + d] = (s, d)
    
    # Check if target is achievable
    if dp[target] == INF:
        # Should not happen, but fallback
        raise ValueError("No solution found")
    
    # Reconstruct the subset
    result = []
    s = target
    while s > 0:
        prev_s, d = parent[s]
        result.append(d)
        s = prev_s
    
    return result
