"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, 5041):
        if 5040 % i == 0:
            divisors.append(i)
    
    # Sort divisors in descending order for greedy initialization
    divisors_desc = sorted(divisors, reverse=True)
    
    # First, get a valid solution using greedy as a fallback
    def greedy_solve():
        rem = 4979
        chosen = []
        for d in divisors_desc:
            if d <= rem:
                chosen.append(d)
                rem -= d
            if rem == 0:
                break
        if rem == 0:
            return chosen
        return None
    
    greedy_result = greedy_solve()
    
    # DP approach: find minimum number of divisors summing to 4979
    # dp[s] = (min_count, chosen_divisors) but storing lists is expensive
    # Use BFS by level: at each level k, find all sums reachable with exactly k divisors
    
    # Better: use DP where dp[s] = minimum number of divisors to reach sum s
    # We also need to track the actual divisors used
    
    # Since target is 4979 and we have 60 divisors, let's use a dictionary-based DP
    # dp maps sum -> (count, tuple_of_divisors) but this could be large
    
    # Alternative: BFS from 0, expanding by adding one divisor at a time
    # Track visited sums to avoid reprocessing
    
    target = 4979
    
    # Use a dictionary: sum -> list of divisors used (for the shortest path)
    # BFS: start with {0: []}, then at each step, add each divisor not yet used
    # This is exponential in worst case, but with 60 divisors and target 4979, it might work
    
    # Actually, let's use a smarter DP:
    # dp[s] = minimum number of divisors to achieve sum s
    # We process divisors one by one (0/1 knapsack style)
    
    # Initialize dp with infinity
    # dp[0] = 0
    # For each divisor d, for s from target down to d:
    #   dp[s] = min(dp[s], dp[s-d] + 1)
    
    # But we need to reconstruct the solution. Let's store parent pointers.
    
    # dp[s] = min count to reach sum s
    # parent[s] = the divisor that was last added to reach s
    # prev_sum[s] = the sum before adding that divisor
    
    INF = float('inf')
    dp = [INF] * (target + 1)
    parent_div = [-1] * (target + 1)  # which divisor was added
    parent_sum = [-1] * (target + 1)  # previous sum
    
    dp[0] = 0
    
    # Process each divisor
    # Micro-optimization: local variable access
    dp_local = dp
    pd_local = parent_div
    ps_local = parent_sum
    
    for d in divisors:
        # 0/1 knapsack: iterate from high to low
        for s in range(target, d - 1, -1):
            prev_val = dp_local[s - d]
            if prev_val != INF:
                new_val = prev_val + 1
                if new_val < dp_local[s]:
                    dp_local[s] = new_val
                    pd_local[s] = d
                    ps_local[s] = s - d
    
    if dp[target] == INF:
        # No solution found via DP, fall back to greedy
        if greedy_result is not None:
            return greedy_result
        # Should not happen
        return []
    
    # Reconstruct the solution
    chosen = []
    s = target
    while s > 0:
        d = parent_div[s]
        chosen.append(d)
        s = parent_sum[s]
    
    chosen.reverse()
    
    # Verify
    assert len(set(chosen)) == len(chosen), "Divisors must be distinct"
    assert sum(chosen) == target, "Sum must equal target"
    for d in chosen:
        assert 5040 % d == 0, f"{d} is not a divisor of 5040"
    
    return chosen

# EVOLVE-BLOCK-END
