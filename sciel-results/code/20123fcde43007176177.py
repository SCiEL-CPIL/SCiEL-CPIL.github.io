"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    deadline = time.time() + 50.0  # safety margin before 90s kill
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = [d for d in range(1, n+1) if n % d == 0]
    # Sort in descending order to prefer large divisors (helps find solutions faster)
    divisors.sort(reverse=True)
    
    num_div = len(divisors)
    
    # DP: dp[s] = minimum number of distinct divisors to achieve sum s
    # Initialize with infinity
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    # parent[s] = (prev_sum, divisor_used) to reconstruct solution
    parent = [None] * (target + 1)
    
    # For each divisor, update dp in reverse to avoid using same divisor twice
    for d in divisors:
        if d > target:
            continue
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
    
    if dp[target] == INF:
        # Fallback: return empty or raise
        return []
    
    # Reconstruct the solution
    chosen = []
    s = target
    while s > 0:
        prev_s, d = parent[s]
        chosen.append(d)
        s = prev_s
    
    return chosen
# EVOLVE-BLOCK-END
