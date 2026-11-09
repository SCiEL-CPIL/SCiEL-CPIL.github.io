"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    num_divs = len(divisors)  # 60 divisors
    
    # DP: dp[s] = minimum number of divisors to achieve sum s
    # We'll use a large array indexed by sum
    # Max possible sum with 60 divisors is sum of all divisors
    max_sum = sum(divisors)
    
    # dp[s] = min count to reach sum s, initialized to infinity
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # parent[s] = (prev_sum, divisor_index) for reconstruction
    parent = [(-1, -1)] * (target + 1)
    
    # Process each divisor (0/1 knapsack style)
    # To minimize count, we want to use larger divisors preferentially
    # Sort divisors in descending order to help find solutions with fewer terms
    sorted_divs = sorted(divisors, reverse=True)
    sorted_indices = [divisors.index(d) for d in sorted_divs]
    
    for idx, d in enumerate(sorted_divs):
        # Process in reverse to avoid using same divisor twice
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, idx)
    
    if dp[target] == INF:
        # Fallback to greedy if DP fails (shouldn't happen)
        remainder = target
        chosen = []
        for d in sorted(divisors, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        assert remainder == 0, f"Could not reach target, remainder={remainder}"
        return chosen
    
    # Reconstruct the solution
    chosen_indices = []
    s = target
    while s > 0:
        prev_s, div_idx = parent[s]
        chosen_indices.append(div_idx)
        s = prev_s
    
    # Convert indices back to divisor values
    chosen = [sorted_divs[i] for i in chosen_indices]
    
    # Verify
    assert len(set(chosen)) == len(chosen), "Divisors not distinct"
    assert all(n % d == 0 for d in chosen), "Not all are divisors of 5040"
    assert sum(chosen) == target, f"Sum is {sum(chosen)}, expected {target}"
    assert len(chosen) <= 256, f"Too many divisors: {len(chosen)}"
    
    return chosen

# EVOLVE-BLOCK-END
