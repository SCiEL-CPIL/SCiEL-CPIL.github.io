"""Exact minimum-cardinality subset-sum using bitset BFS on cardinality layers."""
# EVOLVE-BLOCK-START
import time

def construct():
    target = 4979
    n = 5040
    deadline = time.time() + 55.0  # safety margin before 60s kill

    # Generate all distinct positive divisors of n
    divisors = []
    for d in range(1, int(n ** 0.5) + 1):
        if n % d == 0:
            divisors.append(d)
            if d != n // d:
                divisors.append(n // d)
    
    # Filter: only divisors <= target can be part of a sum equal to target
    usable = [d for d in divisors if d <= target]
    usable.sort(reverse=True)  # largest first

    # Bitset BFS by cardinality:
    # layers[k] = bitset where bit s is set if sum s is reachable with exactly k distinct divisors
    # We process divisors one at a time, updating layers in descending k to avoid reusing a divisor.
    
    max_k = len(usable)
    # layers[k] is an integer; bit i set means sum i is achievable with k divisors
    layers = [0] * (max_k + 1)
    layers[0] = 1  # sum 0 with 0 divisors
    
    # For each divisor, update layers from high k to low k (0/1 knapsack style)
    for d in usable:
        # Update from high k to low k to ensure each divisor used at most once
        for k in range(max_k - 1, -1, -1):
            if layers[k] != 0:
                # Shift the bitset by d and OR into layers[k+1]
                shifted = (layers[k] << d) & ((1 << (target + 1)) - 1)
                layers[k + 1] |= shifted
                # Early check: if target is now reachable with k+1 divisors
                if (layers[k + 1] >> target) & 1:
                    # We found a solution with k+1 divisors; since we're processing
                    # divisors in some order, this might not be globally minimal yet.
                    # But we can note it and continue to see if smaller k works.
                    pass
        
        # Time guard
        if time.time() > deadline:
            break

    # Find minimum k such that target is reachable
    for k in range(1, max_k + 1):
        if (layers[k] >> target) & 1:
            # Reconstruct the solution using parent tracking
            # We need to re-run with parent tracking to get the actual divisors
            break
    else:
        return []  # No solution found

    # Re-run DP with parent tracking to reconstruct the solution
    INF = float('inf')
    dp = [INF] * (target + 1)
    parent = [None] * (target + 1)
    dp[0] = 0
    
    for d in usable:
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
    
    if dp[target] == INF:
        return []
    
    solution = []
    current_sum = target
    while current_sum > 0:
        prev_sum, d = parent[current_sum]
        solution.append(d)
        current_sum = prev_sum
    
    return solution
# EVOLVE-BLOCK-END
