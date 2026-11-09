"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    num_divs = len(divisors)
    
    # Sort divisors in descending order to try larger divisors first (heuristic for fewer terms)
    # But we need to be careful with DP ordering. Actually, for 0/1 knapsack minimizing count,
    # the order of processing items doesn't affect correctness, only which solution we find first.
    # Since we're minimizing count, let's just process all.
    
    # Use two arrays: 
    # dp_count[s] = minimum number of divisors to achieve sum s
    # dp_mask[s] = bitmask of divisors used to achieve that minimum count for sum s
    
    dp_count = [float('inf')] * (target + 1)
    dp_mask = [0] * (target + 1)
    dp_count[0] = 0
    dp_mask[0] = 0
    
    for i, d in enumerate(divisors):
        bit = 1 << i
        # Iterate s from target down to d to avoid reusing the same divisor
        for s in range(target, d - 1, -1):
            if dp_count[s - d] != float('inf'):
                new_count = dp_count[s - d] + 1
                new_mask = dp_mask[s - d] | bit
                if new_count < dp_count[s]:
                    dp_count[s] = new_count
                    dp_mask[s] = new_mask
    
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
