"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    num_divs = len(divisors)
    
    # We want to find the minimum number of distinct divisors that sum to target.
    # dp[s] = bitmask of divisors used, or -1 if unreachable.
    # We also store the count separately for faster comparison.
    
    # Initialize dp array with -1 (unreachable)
    dp = [-1] * (target + 1)
    dp[0] = 0  # bitmask 0 means no divisors chosen
    
    # To speed up popcount, we can precompute or use int.bit_count() (Python 3.10+)
    # Let's use a helper that falls back to bin().count('1') if needed
    # But since we're in a modern environment, int.bit_count() should be available.
    # Let's check and use it if possible, otherwise fallback.
    
    # Actually, let's just use bin(x).count('1') for compatibility, but optimize the loop.
    # The main bottleneck is the inner loop calling bin().count() twice.
    # We can store the count alongside the mask.
    
    # Let's use two arrays: dp_mask[s] and dp_count[s]
    dp_mask = [-1] * (target + 1)
    dp_count = [float('inf')] * (target + 1)
    dp_mask[0] = 0
    dp_count[0] = 0
    
    # Iterate through each divisor
    for i, d in enumerate(divisors):
        bit = 1 << i
        # Iterate s from target-d down to 0
        # We only update if dp_mask[s] is not -1
        # To avoid using the same divisor twice, we go in reverse order of s.
        for s in range(target - d, -1, -1):
            if dp_mask[s] != -1:
                new_mask = dp_mask[s] | bit
                new_count = dp_count[s] + 1
                
                if new_count < dp_count[s + d]:
                    dp_mask[s + d] = new_mask
                    dp_count[s + d] = new_count
    
    if dp_mask[target] == -1:
        # Should not happen for practical numbers, but just in case
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
