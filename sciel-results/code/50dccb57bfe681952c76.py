"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt
from itertools import combinations

P = {'n': 256}

def _valid_set(S):
    """Check that for every three distinct members a,b,c, a does not divide b*c."""
    lst = list(S)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(n):
            if j == i:
                continue
            b = lst[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                c = lst[k]
                if (b * c) % a == 0:
                    return False
    return True

def _fast_check_add(S, x):
    """Check if adding x to set S keeps validity. S is a set of ints not containing x."""
    lst = list(S)
    n = len(lst)
    # Check x as 'a': for all b,c in S distinct, x | b*c must be false
    for i in range(n):
        b = lst[i]
        for j in range(n):
            if j == i:
                continue
            c = lst[j]
            if (b * c) % x == 0:
                return False
    # Check x as 'b' or 'c': for all a in S, and some other d in S, a | x*d must be false
    for i in range(n):
        a = lst[i]
        for j in range(n):
            if j == i:
                continue
            d = lst[j]
            if (x * d) % a == 0:
                return False
    return True

def construct():
    deadline = time.time() + 80.0
    n = P['n']
    
    # Precompute primes
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(n) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    primes = [i for i in range(2, n+1) if is_prime[i]]
    
    # Precompute prime factors for all numbers
    pf = [set() for _ in range(n + 1)]
    for p in primes:
        for j in range(p, n + 1, p):
            pf[j].add(p)
    
    composites = [x for x in range(2, n+1) if not is_prime[x]]
    
    # Strategy: For each subset R of small primes to exclude,
    # S = (primes \ R) ∪ {c composite : pf[c] ⊆ R}
    # Then verify and do local search.
    
    # Focus on small primes for R enumeration (primes up to ~30)
    small_primes = [p for p in primes if p <= 30]
    best_set = set(primes)
    best_size = len(primes)
    
    # First, try the all-primes baseline
    if _valid_set(best_set):
        pass  # It's valid
    else:
        # Shouldn't happen, but fallback
        best_set = set()
        best_size = 0
    
    # Enumerate subsets of small_primes as R
    # For small sets, enumerate all; for larger, use greedy
    sp_len = len(small_primes)
    
    # Try all subsets up to size 4 (manageable)
    for r_size in range(0, min(sp_len + 1, 5)):
        if time.time() > deadline - 15:
            break
        for R_tuple in combinations(small_primes, r_size):
            R = set(R_tuple)
            S = set()
            # Add primes not in R
            for p in primes:
                if p not in R:
                    S.add(p)
            # Add composites whose prime factors are all in R
            for c in composites:
                if pf[c] and pf[c].issubset(R):
                    S.add(c)
            
            if len(S) > best_size:
                # Verify validity
                if _valid_set(S):
                    best_set = S
                    best_size = len(S)
    
    # Local search refinement on best_set
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try adding each composite not in best_set
            for x in composites:
                if x in best_set:
                    continue
                if _fast_check_add(best_set, x):
                    best_set.add(x)
                    best_size = len(best_set)
                    improved = True
            # Try swapping: remove one element, add another
            for elem in list(best_set):
                if time.time() > deadline - 10:
                    break
                trial = set(best_set)
                trial.remove(elem)
                added = False
                for x in composites:
                    if x in trial or x == elem:
                        continue
                    if _fast_check_add(trial, x):
                        trial.add(x)
                        added = True
                if len(trial) > len(best_set):
                    best_set = trial
                    best_size = len(trial)
                    improved = True
    
    # Final verification
    if not _valid_set(best_set):
        # Fallback to primes
        best_set = set(primes)
    
    result = [int(x) for x in sorted(best_set)]
    return result
# EVOLVE-BLOCK-END
