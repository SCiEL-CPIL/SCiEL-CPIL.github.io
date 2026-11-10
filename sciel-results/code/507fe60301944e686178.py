"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

# Precompute prime factorizations for all numbers in [2, 40]
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _factorize(num):
    """Return dict of prime->exponent for num."""
    factors = {}
    x = num
    for p in _PRIMES:
        if x % p == 0:
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            factors[p] = cnt
    return factors

# Precompute factorizations
_FACTS = {n: _factorize(n) for n in range(2, 41)}

def _has_distinct_subset_sums(multiset):
    """Check if all subset sums of the multiset are distinct."""
    # multiset is a list of integers
    k = len(multiset)
    if k == 0:
        return True
    sums = set()
    for mask in range(1 << k):
        s = 0
        for i in range(k):
            if mask & (1 << i):
                s += multiset[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _greedy_build_fast():
    """Greedy construction using the decomposed property."""
    # Start with all primes
    current = list(_PRIMES)
    used = set(current)
    
    # Track exponents for each prime
    # exps[p] is a list of exponents of p in the current numbers
    exps = {p: [] for p in _PRIMES}
    for p in _PRIMES:
        exps[p].append(1)
        
    candidates = [n for n in range(4, 41) if n not in used]
    
    # Heuristic: sort candidates by number of prime factors or something
    # Prefer numbers with fewer prime factors? Or larger?
    # Let's try sorting by value
    candidates.sort()
    
    for cand in candidates:
        if len(current) >= 18:
            break
        
        # Check if adding cand is valid
        valid = True
        facts = _FACTS[cand]
        
        # Temporarily update exps to check
        new_exps = {}
        for p in _PRIMES:
            new_exps[p] = exps[p][:]
            
        for p, e in facts.items():
            new_exps[p].append(e)
            
        for p in _PRIMES:
            if not _has_distinct_subset_sums(new_exps[p]):
                valid = False
                break
                
        if valid:
            current.append(cand)
            used.add(cand)
            for p, e in facts.items():
                exps[p].append(e)
                
    return current

def construct():
    deadline = time.time() + 50
    
    result = _greedy_build_fast()
    
    # If greedy didn't reach 18, try a simple backtracking on the "valid" candidates
    # But greedy is likely sufficient for this structure.
    # If we need more, we can try different orderings.
    
    if len(result) < 18:
        # Try reversing order or other heuristics
        pass
        
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
