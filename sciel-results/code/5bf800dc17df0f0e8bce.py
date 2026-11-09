"""Direct construction for 'Large sets with distinct subset products'."""

import time
from itertools import combinations

DEADLINE = 82.0  # wall-clock seconds (90s limit, safety margin)
T_START = time.time()

def construct():
    global DEADLINE, T_START
    deadline = DEADLINE
    if deadline < 0:
        deadline = time.time() + 82.0

    def time_left():
        return deadline - time.time()

    # Precompute a known good set of 17 elements
    # Strategy: Use prime powers carefully to avoid subset product collisions.
    # For distinct subset products, we need that the mapping from subset to product is injective.
    # This is guaranteed if we can assign each element a unique "signature" in terms of prime exponents
    # such that no two different subsets have the same exponent vector.
    
    # A safe approach: use numbers that are powers of distinct primes, plus a few carefully chosen composites.
    # Actually, let's use a greedy approach with verification.
    
    # Start with all primes (12 elements) - always valid
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Try to add more elements from remaining numbers in [2,40]
    # Candidate numbers not already primes:
    all_numbers = list(range(2, 41))
    prime_set = set(primes)
    candidates = [x for x in all_numbers if x not in prime_set]
    
    # Sort candidates by "complexity" - prefer those with fewer/smaller prime factors
    def prime_factor_complexity(n):
        """Return a tuple that helps us prefer 'simpler' numbers"""
        factors = []
        d = 2
        tmp = n
        while d * d <= tmp:
            while tmp % d == 0:
                factors.append(d)
                tmp //= d
            d += 1
        if tmp > 1:
            factors.append(tmp)
        return (len(factors), sum(factors))
    
    candidates.sort(key=prime_factor_complexity)
    
    best_set = list(primes)
    
    # Greedily try to add candidates
    for cand in candidates:
        if time_left() < 5.0:
            break
        test_set = best_set + [cand]
        n = len(test_set)
        
        # Verify distinct subset products using a set of products
        # For n up to ~18, 2^n can be up to 262144, which is manageable
        prods = set()
        valid = True
        for mask in range(1 << n):
            prod = 1
            for i in range(n):
                if mask & (1 << i):
                    prod *= test_set[i]
            if prod in prods:
                valid = False
                break
            prods.add(prod)
        
        if valid:
            best_set = test_set
    
    # Verify final result
    n = len(best_set)
    prods = set()
    for mask in range(1 << n):
        prod = 1
        for i in range(n):
            if mask & (1 << i):
                prod *= best_set[i]
        if prod in prods:
            # Should not happen, but fallback to primes
            best_set = primes
            break
        prods.add(prod)
    
    res = [int(x) for x in best_set]
    assert len(res) == len(set(res)), "duplicates"
    assert all(isinstance(x, int) and not isinstance(x, bool) and 2 <= x <= 40 for x in res)
    
    return list(res)
