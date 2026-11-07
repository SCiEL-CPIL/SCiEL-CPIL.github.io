"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    import math
    from collections import defaultdict
    
    # Precompute sigma(p^k) for small primes
    # sigma(p^k) = 1 + p + p^2 + ... + p^k = (p^(k+1) - 1) / (p - 1)
    
    # Generate candidate numbers as products of prime powers
    # We want a/b ≈ 3/2, so b ≈ 2a/3
    # Let's generate numbers up to a reasonable bound and track their sigma
    
    # Use a sieve-like approach but only for numbers we need
    # Actually, let's precompute sigma for all numbers up to 10^6 first
    # Then for larger numbers, use factorization
    
    MAX_N = 2000000  # Compute sigma up to this bound
    
    sigma = [0] * (MAX_N + 1)
    for i in range(1, MAX_N + 1):
        for j in range(i, MAX_N + 1, i):
            sigma[j] += i
    
    # Group numbers by their sigma value
    sigma_to_numbers = defaultdict(list)
    for n in range(1, MAX_N + 1):
        sigma_to_numbers[sigma[n]].append(n)
    
    best_pair = None
    best_error = float('inf')
    
    # For each sigma value, check pairs in that group
    # We want a/b ≈ 3/2, so a ≈ 1.5 * b
    # For efficiency, sort each group and use two pointers or binary search
    
    for sig_val, nums in sigma_to_numbers.items():
        if len(nums) < 2:
            continue
        nums.sort()
        # For each b in nums, find a in nums such that a/b ≈ 3/2
        # a ≈ 1.5 * b
        # Use binary search
        for i in range(len(nums)):
            b = nums[i]
            target_a = 1.5 * b
            # Binary search for closest to target_a
            lo, hi = 0, len(nums) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < target_a:
                    lo = mid + 1
                else:
                    hi = mid
            # Check lo and lo-1
            for idx in [lo - 1, lo]:
                if 0 <= idx < len(nums):
                    a = nums[idx]
                    if a == b:
                        continue
                    error = abs(a / b - 3 / 2)
                    if error < best_error:
                        best_error = error
                        best_pair = (a, b)
    
    if best_pair is None:
        return [122, 80]
    
    a, b = best_pair
    # Ensure a and b are positive integers <= 10000000
    if a <= 0 or b <= 0 or a > 10000000 or b > 10000000:
        return [122, 80]
    
    return [int(a), int(b)]

# EVOLVE-BLOCK-END
