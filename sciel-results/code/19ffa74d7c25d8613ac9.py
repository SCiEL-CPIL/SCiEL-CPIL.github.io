"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from itertools import combinations

P = {'n': 256}

def _is_prime(n):
    if n < 2:
        return False
    for d in range(2, isqrt(n) + 1):
        if n % d == 0:
            return False
    return True

def _valid_set(s):
    """Check if set s satisfies the constraint."""
    lst = list(s)
    for i in range(len(lst)):
        a = lst[i]
        for j in range(i+1, len(lst)):
            b = lst[j]
            for k in range(j+1, len(lst)):
                c = lst[k]
                if (b * c) % a == 0:
                    return False
    return True

def construct():
    deadline = time.time() + 85  # 5s safety margin
    
    n = P['n']
    candidates = list(range(2, n + 1))
    
    # Precompute: for each a, which pairs (b,c) have a | b*c
    # Start with all candidates, greedily remove problematic ones
    
    # First, identify "safe" numbers: those not equal to p*q for distinct primes p,q
    primes = [p for p in candidates if _is_prime(p)]
    prime_set = set(primes)
    
    # A number x is "dangerous" if x = p*q for distinct primes p,q both in [2,256]
    # Since all primes in 2..256 are candidates, any such p*q is dangerous
    dangerous = set()
    for i, p in enumerate(primes):
        for q in primes[i+1:]:
            pq = p * q
            if pq <= n:
                dangerous.add(pq)
    
    # Start with all non-dangerous numbers
    current_set = set(x for x in candidates if x not in dangerous)
    
    # Verify and fix if needed
    if not _valid_set(current_set):
        # Greedy removal
        changed = True
        while changed and time.time() < deadline:
            changed = False
            lst = sorted(current_set)
            # Find a violating triple
            for i in range(len(lst)):
                a = lst[i]
                for j in range(i+1, len(lst)):
                    b = lst[j]
                    for k in range(j+1, len(lst)):
                        c = lst[k]
                        if (b * c) % a == 0:
                            # Remove the most "connected" element
                            # Simple heuristic: remove a
                            current_set.remove(a)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
            if not _valid_set(current_set) and not changed:
                # Try removing other elements
                for elem in sorted(current_set, reverse=True):
                    if time.time() > deadline:
                        break
                    trial = current_set - {elem}
                    if _valid_set(trial):
                        current_set = trial
                        changed = True
                        break
    
    result = sorted(current_set)
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
