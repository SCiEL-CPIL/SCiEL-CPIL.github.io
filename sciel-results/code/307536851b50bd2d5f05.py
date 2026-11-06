"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    n = 5040
    target = 4979
    
    # Generate all positive divisors of 5040
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    divisors.sort()
    
    # DP array: dp[s] = minimum number of divisors to sum to s
    # Initialize with infinity
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # choice[s] = the divisor used to reach sum s in the optimal path
    choice = [None] * (target + 1)
    
    # 0/1 Knapsack style DP
    for d in divisors:
        # Iterate backwards to ensure each divisor is used at most once
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                choice[s] = d
                
        # Check time, though this loop is very fast
        if time.time() > deadline:
            break
            
    if dp[target] == INF:
        # Fallback: This should not happen for this specific instance
        # Return an empty list or a greedy attempt if DP fails
        return []
        
    # Backtrack to find the divisors
    result = []
    current_sum = target
    while current_sum > 0:
        d = choice[current_sum]
        if d is None:
            # Should not happen if dp[target] is valid
            break
        result.append(d)
        current_sum -= d
        
    return result
# EVOLVE-BLOCK-END
