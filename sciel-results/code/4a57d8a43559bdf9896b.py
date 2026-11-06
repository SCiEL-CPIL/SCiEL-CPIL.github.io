"""Shortest-path search for minimum number of distinct divisors of 5040 summing to 4979."""
import time
from heapq import heappush, heappop

N = 5040
TARGET = 4979
DEADLINE = 55.0  # safety margin before 60s kill


def _divisors_of(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)


def construct():
    start_time = time.time()
    
    # Precompute all divisors of 5040
    divisors = _divisors_of(N)
    div_set = set(divisors)
    
    # State: (cost, sum_used, frozenset_of_used_divisors)
    # We want to minimize cost (number of divisors) to reach exactly TARGET.
    # Dijkstra: cost = number of divisors used.
    # State space: (sum, frozenset) is too large. Instead, use (sum, mask) where mask tracks used divisors.
    # But 256 divisors -> 2^256 states is impossible.
    
    # Alternative: BFS by number of divisors k=1,2,3,...
    # For each k, check if there exist k distinct divisors summing to TARGET.
    # This is a subset-sum with cardinality constraint.
    # Use dynamic programming: dp[s] = set of masks or just track reachable sums with min count.
    
    # Better approach: DP where dp[s] = minimum number of distinct divisors needed to sum to s.
    # But we need to ensure distinctness. Standard knapsack doesn't handle distinctness easily.
    
    # Since we have at most 256 divisors, and TARGET=4979, we can do:
    # dp[s] = bitmask of which divisors were used? No, too large.
    
    # Let's use a different approach: 
    # dp[s] = set of tuples (count, frozenset) - still too large.
    
    # Practical approach: 
    # The divisors of 5040 are: let's list them.
    # 5040 = 2^4 * 3^2 * 5 * 7
    # Number of divisors = (4+1)(2+1)(1+1)(1+1) = 5*3*2*2 = 60 divisors.
    # So we have 60 divisors to choose from.
    
    # We can do DP: dp[s] = minimum number of divisors to reach sum s, using a bitset to track which divisors are used.
    # But 60 divisors -> 2^60 is too large.
    
    # Alternative: meet-in-the-middle. Split 60 divisors into two groups of 30.
    # For each group, compute all possible sums with their corresponding count and mask.
    # Then combine.
    
    # Actually, let's just do a BFS/A* search with state = (sum, used_mask) where used_mask is an integer bitmask of 60 bits.
    # 2^60 is too large, but we'll prune heavily because sum must be <= 4979.
    
    # Let's try a simpler DP: 
    # dp[s] = set of used-mask values that can achieve sum s.
    # But this can blow up.
    
    # Given the small number of divisors (60) and target (4979), let's try:
    # For k = 1, 2, 3, ...: check if any k distinct divisors sum to TARGET.
    # Use itertools.combinations for small k, but that's expensive for large k.
    
    # Better: use DP with dp[s] = minimum count, and store one witness path.
    # To handle distinctness, process divisors one by one (0/1 knapsack style).
    
    # dp[s] = min number of divisors to reach sum s
    # prev[s] = (prev_sum, divisor_used) for reconstruction
    # Process each divisor once (0/1 knapsack).
    
    INF = float('inf')
    max_sum = TARGET
    dp = [INF] * (max_sum + 1)
    prev_sum = [-1] * (max_sum + 1)
    prev_div = [-1] * (max_sum + 1)
    dp[0] = 0
    
    for d in divisors:
        # 0/1 knapsack: iterate backwards
        if time.time() - start_time > DEADLINE:
            break
        for s in range(max_sum, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                prev_sum[s] = s - d
                prev_div[s] = d
    
    if dp[TARGET] == INF:
        # Should not happen, but return empty as fallback
        return []
    
    # Reconstruct the solution
    chosen = []
    s = TARGET
    while s > 0:
        d = prev_div[s]
        chosen.append(d)
        s = prev_sum[s]
    
    # Verify
    assert len(set(chosen)) == len(chosen), "Duplicate divisors found"
    assert all(d in div_set for d in chosen), "Non-divisor found"
    assert sum(chosen) == TARGET, "Sum mismatch"
    assert len(chosen) <= 256, "Too many divisors"
    
    return chosen
