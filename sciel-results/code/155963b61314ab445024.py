"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, n + 1):
        if n % i == 0:
            divisors.append(i)
    divisors.sort(reverse=True)
    divisor_set = set(divisors)
    ndiv = len(divisors)
    
    # Precompute prefix sums for pruning (largest to smallest)
    # prefix_sum[k] = sum of divisors[0:k] (the k largest divisors)
    prefix_sum = [0] * (ndiv + 1)
    for i in range(ndiv):
        prefix_sum[i + 1] = prefix_sum[i] + divisors[i]
    
    # Dijkstra's algorithm to find minimum number of divisors summing to target
    # State: (remaining_sum, next_index) -> min count
    # We want to minimize the number of divisors used
    
    # dp[remaining_sum] = min number of divisors needed to achieve that sum
    # But we need to track which divisors have been used to ensure distinctness
    # Since we process divisors in order and each can be used at most once,
    # we use a 0/1 knapsack style DP with reconstruction
    
    INF = 10**9
    dp = [INF] * (target + 1)
    dp[0] = 0
    parent = [(-1, -1)] * (target + 1)  # parent[s] = (prev_sum, divisor_index)
    
    for d_idx, d in enumerate(divisors):
        # Update in reverse to ensure each divisor is used at most once
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d_idx)
        
        # Deadline check
        if time.time() > deadline - 5:
            break
    
    min_count = dp[target]
    
    if min_count == INF:
        # Fallback: try a greedy approach
        # Sort divisors descending, pick largest that fits
        result = []
        remaining = target
        for d in divisors:
            if d <= remaining:
                result.append(d)
                remaining -= d
                if remaining == 0:
                    break
        if remaining == 0:
            return result
        return []
    
    # Reconstruct the solution
    result = []
    s = target
    while s > 0:
        prev_s, d_idx = parent[s]
        result.append(divisors[d_idx])
        s = prev_s
    
    return result
# EVOLVE-BLOCK-END
