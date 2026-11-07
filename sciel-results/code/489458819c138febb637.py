"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, isqrt(n) + 1, 2):
        if n % i == 0:
            return False
    return True

def _validate(s):
    """Check that for every three distinct a,b,c in s, a does not divide b*c."""
    s_list = list(s)
    n = len(s_list)
    for i in range(n):
        a = s_list[i]
        for j in range(n):
            if j == i:
                continue
            b = s_list[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                c = s_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def construct():
    deadline = time.time() + 50.0  # safety margin
    
    # Start with all primes in [2, 256]
    primes = [a for a in range(2, 257) if _is_prime(a)]
    
    # Verify primes-only set is valid
    if not _validate(primes):
        # Fallback: just return primes (shouldn't happen)
        return primes
    
    # Try to add composite numbers one by one
    composites = [a for a in range(2, 257) if not _is_prime(a)]
    composites.sort(reverse=True)  # Try larger composites first? Or smaller?
    
    # Actually, let's try all composites in order and greedily add if valid
    result = list(primes)
    
    for c in composites:
        if time.time() > deadline - 5.0:
            break
        candidate = result + [c]
        if _validate(candidate):
            result = candidate
    
    # Try local search: swap operations
    # For each element in result, try replacing it with a non-member
    non_members = [a for a in range(2, 257) if a not in result]
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(len(result)):
            for nm in non_members:
                if time.time() > deadline - 5.0:
                    break
                candidate = result.copy()
                candidate[i] = nm
                if _validate(candidate):
                    # Check if this actually improves (it doesn't change size, but might enable further improvements)
                    # Actually, swaps don't increase size, so skip unless we can then add more
                    pass
            if time.time() > deadline - 5.0:
                break
        # Try adding after swaps is complex; for now, just try direct additions again
        # Try each non-member addition
        for nm in non_members:
            if time.time() > deadline - 5.0:
                break
            if nm not in result:
                candidate = result + [nm]
                if _validate(candidate):
                    result = candidate
                    non_members.remove(nm)
                    improved = True
                    break
    
    # Final verification
    if not _validate(result):
        # Fallback to primes only
        return primes
    
    return list(result)
# EVOLVE-BLOCK-END
