"""
Optimized minimum-cardinality subset-sum solver for divisors of 5040.
Uses a single-pass 0/1 Knapsack Dynamic Programming approach to minimize
the number of distinct divisors summing to the target.
"""
import time

def construct():
    target = 4979
    n = 5040
    deadline = time.time() + 55.0  # Safety margin

    # 1. Generate all distinct positive divisors of n
    divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
        i += 1
    
    # Filter divisors: only those <= target are useful
    usable = [d for d in divisors if d <= target]
    
    # Sort in descending order? 
    # For standard DP, order doesn't change the result, but descending 
    # might allow early pruning in some heuristic variants. 
    # However, for strict min-count DP, we process all.
    # We don't strictly need to sort for DP correctness, but it's good for consistency.
    usable.sort(reverse=True)

    # 2. Dynamic Programming
    # dp[s] stores the minimum number of divisors needed to sum to s.
    # If dp[s] is infinity, it's unreachable.
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # parent[s] stores (prev_sum, divisor_used) to reconstruct the path
    parent = [None] * (target + 1)

    # Process each divisor
    for d in usable:
        # Update from high to low to ensure 0/1 property (each divisor used at most once)
        # We only care about sums up to 'target'
        for s in range(target, d - 1, -1):
            # If we can reach s-d with fewer items, and adding d gives a better total
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
        
        # Time guard (unlikely to be needed for this size, but required by contract)
        if time.time() > deadline:
            break

    # 3. Check if solution exists
    if dp[target] == INF:
        return []
    
    # 4. Reconstruct the solution
    solution = []
    current_sum = target
    while current_sum > 0:
        prev_sum, d_used = parent[current_sum]
        solution.append(d_used)
        current_sum = prev_sum
    
    # The solution is a list of distinct divisors
    return solution
