"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    num_divs = len(divisors)
    
    # We want to find the minimum number of distinct divisors that sum to target.
    # Use DP: dp[s] = (count, bitmask) for minimum count to achieve sum s.
    # Since 5040 has 60 divisors, bitmask fits in a Python int.
    # We process divisors one by one (0/1 knapsack style) to ensure each is used at most once.
    
    # Initialize: dp[s] = None if unreachable, else (count, bitmask)
    # Use two arrays: dp_count[s] and dp_mask[s]
    dp_count = [float('inf')] * (target + 1)
    dp_mask = [0] * (target + 1)
    dp_count[0] = 0
    dp_mask[0] = 0
    
    # Process each divisor
    for i, d in enumerate(divisors):
        bit = 1 << i
        # Iterate s from target down to 0 to avoid reusing the same divisor
        for s in range(target - d, -1, -1):
            if dp_count[s] != float('inf'):
                new_count = dp_count[s] + 1
                new_mask = dp_mask[s] | bit
                if new_count < dp_count[s + d]:
                    dp_count[s + d] = new_count
                    dp_mask[s + d] = new_mask
    
    if dp_count[target] == float('inf'):
        raise ValueError("No solution found")
    
    # Reconstruct the divisors from the bitmask
    result_mask = dp_mask[target]
    selected_divisors = []
    for i in range(num_divs):
        if result_mask & (1 << i):
            selected_divisors.append(divisors[i])
    
    # Verify sum and distinctness
    assert len(selected_divisors) == len(set(selected_divisors))
    assert sum(selected_divisors) == target
    assert all(n % d == 0 for d in selected_divisors)
    
    return selected_divisors
# EVOLVE-BLOCK-END
