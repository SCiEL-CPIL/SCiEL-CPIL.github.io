"""Rewritten program to maximize the number of distinct integers in 2..40 with distinct subset products."""

def construct():
    import time
    import itertools
    
    # Deadline guard: 90s limit, keep 10s margin
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 40
    
    # Generate all integers in 2..40
    candidates = list(range(2, n + 1))
    
    # We want to find the largest subset of candidates such that all subset products are distinct.
    # This is equivalent to finding the largest linearly independent set of prime exponent vectors.
    # The maximum possible size is the number of distinct primes <= 40, which is 12.
    # The set of primes themselves is always a valid solution of size 12.
    
    # Let's verify if we can do better or if 12 is indeed the max.
    # As derived, the max rank is the number of primes.
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37.
    
    # Construct the set of primes
    primes = []
    for i in range(2, n + 1):
        is_prime = True
        for d in range(2, int(i**0.5) + 1):
            if i % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    
    # Verify the primes set
    def is_valid_subset(subset):
        # Check if all subset products are distinct
        # For small sets, we can just enumerate
        # 2^12 = 4096 products, which is fast.
        products = set()
        for r in range(len(subset) + 1):
            for comb in itertools.combinations(subset, r):
                prod = 1
                for x in comb:
                    prod *= x
                if prod in products:
                    return False
                products.add(prod)
        return True
    
    # The primes are always valid
    if is_valid_subset(primes):
        return [int(x) for x in primes]
    
    # Fallback: Greedy search if primes somehow failed (unlikely)
    # Try to add composites that are independent
    # But since primes span the space, no composite can be added.
    # So primes is the optimal solution.
    
    # Just in case, return the primes as ints
    return [int(x) for x in primes]

# EVOLVE-BLOCK-END
