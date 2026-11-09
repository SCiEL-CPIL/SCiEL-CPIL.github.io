"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

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
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # parent[s] = (prev_sum, divisor_index) for reconstruction
    parent = [(-1, -1)] * (target + 1)
    
    # Process each divisor (0/1 knapsack style)
    for idx, d in enumerate(divisors):
        # Process in reverse to avoid using same divisor twice
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, idx)
    
    if dp[target] == INF:
        raise ValueError(f"Could not reach target={target}")
    
    # Reconstruct the solution
    chosen = []
    s = target
    while s > 0:
        prev_s, div_idx = parent[s]
        chosen.append(divisors[div_idx])
        s = prev_s
    
    # Verify
    assert len(set(chosen)) == len(chosen), "Divisors not distinct"
    assert all(n % d == 0 for d in chosen), "Not all are divisors of 5040"
    assert sum(chosen) == target, f"Sum is {sum(chosen)}, expected {target}"
    assert len(chosen) <= 256, f"Too many divisors: {len(chosen)}"
    
    return chosen

# EVOLVE-BLOCK-END
