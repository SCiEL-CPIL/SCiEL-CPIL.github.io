"""Rewritten program: Exact minimum-cardinality subset-sum using Dynamic Programming."""
# EVOLVE-BLOCK-START
import time

def construct():
    target = 4979
    n = 5040
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # Step 1: Generate all distinct positive divisors of n
    divisors = []
    for d in range(1, int(n**0.5) + 1):
        if n % d == 0:
            divisors.append(d)
            if d != n // d:
                divisors.append(n // d)
    
    # Filter out divisors larger than target
    divisors = [d for d in divisors if d <= target]
    
    # Step 2: Dynamic Programming to find minimum number of divisors summing to target
    # dp[s] stores the minimum number of divisors needed to sum to s
    # parent[s] stores (prev_sum, divisor_used) to reconstruct the solution
    
    INF = float('inf')
    dp = [INF] * (target + 1)
    parent = [None] * (target + 1)
    
    dp[0] = 0
    
    # Iterate through each divisor
    # Note: We don't need to sort divisors for the DP correctness, 
    # but processing them in any order works for 0/1 knapsack style DP.
    for d in divisors:
        if time.time() > deadline:
            break
        
        # Update dp from high to low to ensure each divisor is used at most once
        # We only care about sums up to target
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
    
    # Step 3: Reconstruct the solution
    if dp[target] == INF:
        return [] # No solution found (should not happen for practical numbers)
    
    solution = []
    current_sum = target
    while current_sum > 0:
        prev_sum, d = parent[current_sum]
        solution.append(d)
        current_sum = prev_sum
    
    return solution
# EVOLVE-BLOCK-END
