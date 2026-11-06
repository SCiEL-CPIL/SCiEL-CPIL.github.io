"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Generate all divisors of 5040
    n = 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort(reverse=True)
    target = 4979
    
    # Use DP to find minimum number of distinct divisors summing to target
    # dp[s] = minimum number of divisors to achieve sum s
    # We track which divisors are used via backtracking
    
    # First, check if greedy works and gives a valid solution
    remainder = target
    greedy_chosen = []
    for d in divisors:
        if d <= remainder:
            greedy_chosen.append(d)
            remainder -= d
        if remainder == 0:
            break
    
    greedy_count = len(greedy_chosen) if remainder == 0 else float('inf')
    
    # Now do exact DP to find the true minimum
    # dp[s] = (min_count, frozenset_of_indices) is too heavy
    # Instead: dp[s] = min count, and parent pointers for reconstruction
    
    # Since we have at most 60 divisors and target ~5000,
    # we can do a BFS/DP over (sum, count) states
    
    # Better approach: for each possible count k, check if we can achieve target
    # with exactly k divisors using DP
    
    # dp[k][s] = True if we can achieve sum s with exactly k divisors
    # But this is O(60 * 5000 * 60) which might be slow
    
    # Alternative: meet-in-the-middle or branch and bound
    
    # Let's use a smarter DP:
    # Process divisors one by one, dp[s] = minimum divisors to reach sum s
    # This is a 0-1 knapsack variant
    
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    # parent[s] = (prev_sum, divisor_used) for reconstruction
    parent = [None] * (target + 1)
    
    for d in divisors:
        # Update in reverse to avoid using same divisor twice
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
    
    # Check if we can achieve target
    if dp[target] == INF:
        # Should not happen for this problem, but fallback to greedy
        if remainder == 0:
            greedy_chosen.sort()
            return greedy_chosen
        else:
            # Return empty or best effort
            return []
    
    # Reconstruct the solution
    chosen = []
    s = target
    while s > 0:
        prev_s, d = parent[s]
        chosen.append(d)
        s = prev_s
    
    chosen.sort()
    
    # Verify the solution
    assert len(set(chosen)) == len(chosen), "Divisors must be distinct"
    assert sum(chosen) == target, f"Sum {sum(chosen)} != target {target}"
    for d in chosen:
        assert n % d == 0, f"{d} is not a divisor of {n}"
    
    return chosen
# EVOLVE-BLOCK-END
