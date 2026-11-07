"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

def construct():
    N = 256
    candidates = list(range(1, N + 1))
    
    # Precompute prime factors for each number
    def prime_factors(n):
        if n == 1:
            return set()
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    # Precompute factor sets
    factor_sets = {i: prime_factors(i) for i in candidates}
    
    # Greedy algorithm:
    # Sort candidates by value descending.
    # Maintain a set of "used" prime factors.
    # If a candidate's prime factors are disjoint from the used set, select it.
    sorted_candidates = sorted(candidates, key=lambda x: -x)
    
    used_primes = set()
    selected = []
    
    for v in sorted_candidates:
        factors = factor_sets[v]
        if not factors:
            # v is 1, always compatible
            selected.append(v)
        else:
            # Check if any factor is already used
            if not used_primes.intersection(factors):
                selected.append(v)
                used_primes.update(factors)
    
    return sorted(selected)

# EVOLVE-BLOCK-END
