"""Combined program: Fast integer-product greedy with random restarts."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

@lru_cache(maxsize=None)
def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True

@lru_cache(maxsize=None)
def _prime_factors(n):
    """Return sorted tuple of prime factors of n (with multiplicity)."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return tuple(factors)

def _subset_products(S):
    """Compute all 2^k subset products of S."""
    prods = [1]
    for a in S:
        prods = prods + [x * a for x in prods]
    return prods

def _has_distinct_subset_products(S):
    """Check that all 2^k subset products of S are distinct."""
    if len(S) == 0:
        return True
    prods = _subset_products(S)
    return len(prods) == len(set(prods))

def _can_add(S, candidate):
    """
    Check if candidate can be added to S while maintaining distinct subset products.
    Uses incremental check: new products are old_products * candidate.
    Valid iff no new product collides with existing or other new products.
    """
    if candidate in S:
        return False
    # Quick reject: if candidate's prime factors are all in S as elements
    cand_factors = set(_prime_factors(candidate))
    s_set = set(S)
    if cand_factors.issubset(s_set):
        return False
    
    # Incremental check
    existing_prods = set(_subset_products(S))
    new_prods = [p * candidate for p in existing_prods]
    # Check for collisions among new products
    if len(new_prods) != len(set(new_prods)):
        return False
    # Check for collisions between new and existing
    for np in new_prods:
        if np in existing_prods:
            return False
    return True

def _greedy_build(start_set, candidates, order_func=None, max_size=18, deadline=None):
    """
    Greedily build a set starting from start_set, adding candidates in given order.
    """
    S = list(start_set)
    
    if not _has_distinct_subset_products(S):
        return []
    
    if order_func:
        sorted_cands = sorted(candidates, key=order_func)
    else:
        sorted_cands = list(candidates)
    
    for c in sorted_cands:
        if len(S) >= max_size:
            break
        if deadline and time.time() > deadline:
            break
        if c in S:
            continue
        if _can_add(S, c):
            S.append(c)
    
    return S

def _random_greedy(all_candidates, max_size=18, deadline=None):
    """
    Randomly order candidates and greedily add them.
    """
    cands = list(all_candidates)
    random.shuffle(cands)
    return _greedy_build([], cands, order_func=None, max_size=max_size, deadline=deadline)

def _local_search_improve(S, all_candidates, max_iter=20, deadline=None):
    """
    Try to improve S by removing one element and adding a different one.
    """
    current = list(S)
    best = list(current)
    all_set = set(all_candidates)
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(len(current)):
            if deadline and time.time() > deadline:
                break
            trial = current[:i] + current[i+1:]
            for cand in all_candidates:
                if cand in trial:
                    continue
                if _can_add(trial, cand):
                    new_trial = trial + [cand]
                    if len(new_trial) > len(best):
                        best = new_trial
                        current = new_trial
                        improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best

def _build_max_set():
    """
    Try multiple strategies to find max set.
    """
    primes = [p for p in range(2, 41) if _is_prime(p)]
    composites = [c for c in range(2, 41) if not _is_prime(c)]
    all_candidates = list(range(2, 41))
    
    t0 = time.time()
    deadline = t0 + 80  # Safety margin
    
    best_set = []
    
    # Strategy 1: Greedy from primes with various orders
    orders = [
        ('asc', lambda x: x),
        ('desc', lambda x: -x),
        ('fewest_factors', lambda x: len(_prime_factors(x))),
        ('most_factors', lambda x: -len(_prime_factors(x))),
    ]
    
    for name, order_func in orders:
        if time.time() > deadline:
            break
        S = _greedy_build(primes, composites, order_func=order_func, max_size=18, deadline=deadline)
        if len(S) > len(best_set):
            best_set = S
    
    # Strategy 2: Try removing individual primes
    for i in range(len(primes)):
        if time.time() > deadline:
            break
        psubset = primes[:i] + primes[i+1:]
        for order_func in [lambda x: x, lambda x: -x]:
            S = _greedy_build(psubset, composites, order_func=order_func, max_size=18, deadline=deadline)
            if len(S) > len(best_set):
                best_set = S
    
    # Strategy 3: Random restarts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        S = _random_greedy(all_candidates, max_size=18, deadline=deadline)
        if len(S) > len(best_set):
            best_set = S
    
    # Strategy 4: Local search on best found
    if best_set and time.time() < deadline - 10:
        improved = _local_search_improve(best_set, all_candidates, max_iter=30, deadline=deadline)
        if len(improved) > len(best_set):
            best_set = improved
    
    return best_set

def construct():
    """Return up to 18 distinct integers in 2..40 with all subset products distinct."""
    t0 = time.time()
    
    # Set random seed for reproducibility within a run
    random.seed(42)
    
    # Primary strategy: Multi-strategy greedy + random restarts + local search
    S = _build_max_set()
    
    # Ensure list of Python ints
    S = [int(x) for x in S]
    
    # Trim to max 18 if needed
    if len(S) > 18:
        S = S[:18]
        
    # Final verification
    if time.time() - t0 < 85:
        prods = [1]
        for a in S:
            prods = prods + [x * a for x in prods]
        if len(prods) != len(set(prods)):
            # Fallback: return just primes
            S = [p for p in range(2, 41) if _is_prime(p)]
            
    return S
# EVOLVE-BLOCK-END
