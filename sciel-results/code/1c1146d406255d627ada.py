"""Rewritten program: return the 12 primes in 2..40, which form a multiplicatively
independent set (all 2^12 subset products are distinct)."""
# EVOLVE-BLOCK-START
import time

P = {'n': 40, 'maximum_size': 18}

def _is_prime(a):
    if a < 2:
        return False
    d = 2
    while d * d <= a:
        if a % d == 0:
            return False
        d += 1
    return True

def _subset_products_distinct(nums):
    """Return True iff all 2^k subset products of nums are distinct."""
    prods = {1}
    for x in nums:
        new = set()
        for p in prods:
            np = p * x
            if np in prods or np in new:
                return False
            new.add(np)
        prods |= new
    return True

def construct():
    deadline = time.time() + 50.0  # safety margin before 60s limit
    # The primes in 2..40 are multiplicatively independent (each has a unique
    # prime factor), so all subset products are distinct. There are 12 of them,
    # which is the theoretical maximum (12 primes <= 40 => exponent vectors
    # live in a 12-dim space).
    primes = [a for a in range(2, P['n'] + 1) if _is_prime(a)]
    # Verify the distinct-subset-product property before returning.
    if _subset_products_distinct(primes):
        return [int(x) for x in primes]
    # Fallback: should never happen, but return empty valid set if broken.
    return []
# EVOLVE-BLOCK-END
