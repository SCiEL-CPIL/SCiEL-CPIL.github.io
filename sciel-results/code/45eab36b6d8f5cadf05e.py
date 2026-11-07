"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

_P = {'n': 256}
_DEADLINE = 60.0

def _is_prime(a):
    if a < 2:
        return False
    if a < 4:
        return True
    if a % 2 == 0:
        return False
    i = 3
    while i * i <= a:
        if a % i == 0:
            return False
        i += 2
    return True

def _valid_set(s):
    """Return True iff for every three distinct members a,b,c: a does not divide b*c."""
    L = list(s)
    n = len(L)
    for i in range(n):
        a = L[i]
        for j in range(n):
            if j == i:
                continue
            b = L[j]
            for k in range(j + 1, n):
                if k == i:
                    continue
                c = L[k]
                if (b * c) % a == 0:
                    return False
    return True

def construct():
    t0 = time.time()
    n = _P['n']
    
    # Generate all primes up to n
    primes = [a for a in range(2, n + 1) if _is_prime(a)]
    
    # Verify the prime set is valid
    if not _valid_set(primes):
        # Fallback: greedily build a valid set
        result = []
        for x in range(2, n + 1):
            if time.time() - t0 > _DEADLINE - 5:
                break
            candidate = result + [x]
            if _valid_set(candidate):
                result = candidate
        return [int(x) for x in result]
    
    # Try to see if we can add any composite by removing its prime factors
    # and check if net gain is positive
    best_set = primes[:]
    best_size = len(best_set)
    
    # Try swapping: remove a prime p, add p^k for k>=2
    # This gives net 0, so no improvement, but let's verify
    # Try adding composites whose prime factors are all removed
    # For each composite c, find its prime factors, remove them, add c
    # Check if valid and if size increases
    
    composites = [x for x in range(2, n + 1) if not _is_prime(x)]
    
    for c in composites:
        if time.time() - t0 > _DEADLINE - 10:
            break
        # Find prime factors of c
        factors = set()
        temp = c
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                factors.add(d)
                temp //= d
            d += 1
        if temp > 1:
            factors.add(temp)
        
        # Only consider if all prime factors are in our prime set
        if all(f in best_set for f in factors):
            # Try: remove factors, add c
            new_set = [x for x in best_set if x not in factors] + [c]
            if len(new_set) > best_size:
                if _valid_set(new_set):
                    best_set = new_set
                    best_size = len(best_set)
    
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
