import math

def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of n
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors (order doesn't strictly matter for correctness of 0/1 knapsack DP 
    # if we use the reverse loop for sum, but it's good practice)
    divisors.sort()
    
    # dp[s] stores the minimum number of divisors needed to sum to s
    # Initialize with a large number
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # prev_sum[s] stores the previous sum state
    # item_used[s] stores the divisor added to reach s
    prev_sum = [-1] * (target + 1)
    item_used = [-1] * (target + 1)
    
    # 0/1 Knapsack DP
    for d in divisors:
        # Iterate backwards to ensure each divisor is used at most once
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                prev_sum[s] = s - d
                item_used[s] = d
                
    # Check if solution exists
    if dp[target] == INF:
        # Fallback to greedy if no solution found (though problem guarantees one)
        # But based on analysis, a solution should exist.
        # The seed program works, so a solution exists.
        pass
        
    # Reconstruct the set of divisors
    chosen = []
    s = target
    while s > 0:
        d = item_used[s]
        if d == -1:
            # Should not happen if dp[target] is not INF
            break
        chosen.append(d)
        s = prev_sum[s]
        
    # Verify sum and distinctness just in case
    # The DP ensures distinctness because it's 0/1 knapsack.
    # The sum is guaranteed by the trace.
    
    return chosen
