"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _has_distinct_subset_products(nums):
    """Check if all 2^k subset products are distinct."""
    k = len(nums)
    if k > 20:
        return False
    products = [1]
    for num in nums:
        products = products + [p * num for p in products]
    return len(set(products)) == len(products)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    # The maximum number of multiplicatively independent integers in 2..40
    # is bounded by the number of distinct primes <= 40, which is 12.
    # All 12 primes give distinct subset products.
    # Return all primes in 2..40.
    from math import isqrt
    primes = [a for a in range(2, 41) if all(a % d for d in range(2, isqrt(a) + 1))]
    
    # Verify the result is valid
    if _has_distinct_subset_products(primes):
        return [int(x) for x in primes]
    
    # Fallback: greedy construction
    best = []
    for cand in range(2, 41):
        if time.time() > deadline:
            break
        test = best + [cand]
        if _has_distinct_subset_products(test):
            best = test
    
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
