"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

# Primes up to 40
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
NUM_PRIMES = len(PRIMES)

def _factorize(n):
    """Return exponent vector of n w.r.t. PRIMES."""
    exponents = [0] * NUM_PRIMES
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            exponents[i] += 1
            n //= p
    return tuple(exponents)

# Precompute exponent vectors for all numbers 2..40
_EXP_VECTORS = {}
for n in range(2, 41):
    _EXP_VECTORS[n] = _factorize(n)

def _subset_products_distinct(nums):
    """Check if all 2^k subset products (as exponent vectors) are distinct."""
    k = len(nums)
    if k == 0:
        return True
    vectors = [_EXP_VECTORS[n] for n in nums]
    # Generate all subset sums
    seen = set()
    seen.add(tuple([0] * NUM_PRIMES))  # empty subset
    for mask in range(1, 1 << k):
        s = [0] * NUM_PRIMES
        for i in range(k):
            if mask & (1 << i):
                v = vectors[i]
                for j in range(NUM_PRIMES):
                    s[j] += v[j]
        t = tuple(s)
        if t in seen:
            return False
        seen.add(t)
    return True

def _try_add(nums, candidate):
    """Try to add candidate to nums, return True if still valid."""
    new_nums = nums + [candidate]
    return _subset_products_distinct(new_nums)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s
    
    # Start with all primes
    primes_in_range = [p for p in PRIMES if p <= 40]
    best = list(primes_in_range)
    
    # Greedy: try to add composites
    composites = [n for n in range(2, 41) if n not in primes_in_range]
    for c in composites:
        if _try_add(best, c):
            best.append(c)
    
    # Local search: try swaps to increase size
    improved = True
    while improved:
        improved = False
        current_size = len(best)
        # Try replacing each element with a non-element
        for i in range(current_size):
            for c in range(2, 41):
                if c in best:
                    continue
                new_set = best[:i] + best[i+1:] + [c]
                if _subset_products_distinct(new_set):
                    # Check if this allows us to add more
                    # Try to build up from this new set
                    new_best = list(new_set)
                    for d in range(2, 41):
                        if d not in new_best:
                            if _try_add(new_best, d):
                                new_best.append(d)
                    if len(new_best) > current_size:
                        best = new_best
                        improved = True
                        break
            if improved:
                break
        
        if time.time() > deadline:
            break
    
    # Final greedy pass
    for c in range(2, 41):
        if c not in best:
            if _try_add(best, c):
                best.append(c)
    
    # Return as list of Python ints
    return list(best)
# EVOLVE-BLOCK-END
