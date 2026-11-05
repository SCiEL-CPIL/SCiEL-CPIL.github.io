"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0
    
    n = 5040
    target = 4979
    
    # Generate all distinct positive divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors in ascending order for DP (order doesn't strictly matter for 0/1 knapsack 
    # correctness, but ascending is standard. We just need distinct divisors).
    divisors = list(set(divisors))
    
    # DP approach to find minimum number of divisors summing to target
    # dp[s] = minimum number of divisors to sum to s
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # parent[s] = (prev_sum, divisor_used)
    parent = [(-1, -1)] * (target + 1)
    
    for d in divisors:
        # Update dp in reverse order to ensure each divisor is used at most once
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
        
        # Check deadline periodically
        if time.time() > deadline:
            break
            
    if dp[target] == INF:
        # Fallback to greedy if DP failed (should not happen for this instance)
        divisors.sort(reverse=True)
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
            if remainder == 0:
                break
        if remainder == 0:
            return chosen
        return [1] # Should not reach here
    
    # Reconstruct the solution
    solution = []
    s = target
    while s > 0:
        prev_s, d = parent[s]
        solution.append(d)
        s = prev_s
    
    # Verify solution
    # Distinctness is guaranteed by 0/1 knapsack structure
    # Sum is guaranteed by DP construction
    # Membership: all d are from divisors list
    # Minimize count: DP finds minimum count
    
    return solution

# EVOLVE-BLOCK-END
