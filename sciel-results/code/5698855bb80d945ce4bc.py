"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

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
        for j in range(i + 1, n):
            b = L[j]
            for k in range(j + 1, n):
                c = L[k]
                # Check all 3 choices of which is the divisor
                if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                    return False
    return True

def _check_addition(s, x):
    """Check if adding x to set s keeps it valid. s is a list, x is int not in s."""
    n = len(s)
    for i in range(n):
        a = s[i]
        for j in range(i + 1, n):
            b = s[j]
            # Three distinct: a, b, x
            if (b * x) % a == 0 or (a * x) % b == 0 or (a * b) % x == 0:
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
            if _check_addition(result, x):
                result.append(x)
        return [int(x) for x in result]
    
    best_set = primes[:]
    best_size = len(best_set)
    
    # Try to add composites: for each composite, check if it can be added
    # without violating constraints. If not, try removing some elements and adding it.
    
    composites = [x for x in range(2, n + 1) if not _is_prime(x)]
    
    # First pass: try adding each composite directly
    for c in composites:
        if time.time() - t0 > _DEADLINE - 30:
            break
        if c not in best_set and _check_addition(best_set, c):
            best_set.append(c)
            best_size += 1
    
    # Second pass: try swapping - remove one prime, add a composite
    for c in composites:
        if time.time() - t0 > _DEADLINE - 15:
            break
        if c in best_set:
            continue
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
        
        # Try removing each subset of factors (size 1 or 2) and adding c
        f_list = list(factors)
        # Remove 1 factor
        for f in f_list:
            if f in best_set:
                new_set = [x for x in best_set if x != f] + [c]
                if len(new_set) > best_size and _valid_set(new_set):
                    best_set = new_set
                    best_size = len(best_set)
                    break
        # Remove 2 factors
        if len(f_list) >= 2:
            for i in range(len(f_list)):
                for j in range(i + 1, len(f_list)):
                    f1, f2 = f_list[i], f_list[j]
                    if f1 in best_set and f2 in best_set:
                        new_set = [x for x in best_set if x != f1 and x != f2] + [c]
                        if len(new_set) > best_size and _valid_set(new_set):
                            best_set = new_set
                            best_size = len(best_set)
    
    # Third pass: try greedy addition of remaining candidates
    for x in range(2, n + 1):
        if time.time() - t0 > _DEADLINE - 5:
            break
        if x not in best_set:
            if _check_addition(best_set, x):
                best_set.append(x)
                best_size += 1
    
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
