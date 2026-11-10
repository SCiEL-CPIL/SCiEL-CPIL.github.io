"""Combined program: structural prime-power independence + greedy composite refinement."""
# EVOLVE-BLOCK-START
import time

_DEADLINE = 85  # safety margin before 90s kill

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

def _has_distinct_subset_products(S):
    """Check that all 2^k subset products of S are distinct."""
    k = len(S)
    if k == 0:
        return True
    prods = [1]
    for a in S:
        new_prods = [x * a for x in prods]
        prods.extend(new_prods)
    return len(prods) == len(set(prods))

def _build_candidates():
    """
    Strategy:
    1. Start with all primes <= 40 (12 primes) - guaranteed multiplicatively independent.
    2. Greedily try to add composite numbers (4, 6, 8, 9, 10, ...) in increasing order,
       keeping a composite only if all 2^(k+1) subset products remain distinct.
    3. Cap at 18 elements.
    
    The key insight: primes are the "atoms" of multiplicative independence.
    Composites can be added if their prime-factorization vector is linearly
    independent (over Z) from the vectors of already-selected elements.
    The subset-product check is the exact test.
    """
    primes = [p for p in range(2, 41) if _is_prime(p)]
    composites = [c for c in range(2, 41) if not _is_prime(c)]
    
    S = list(primes)
    
    # Verify initial set (should always pass)
    if not _has_distinct_subset_products(S):
        S = []
    
    # Greedily add composites
    for c in composites:
        if time.time() > _DEADLINE:
            break
        if len(S) >= 18:
            break
        trial = S + [c]
        if _has_distinct_subset_products(trial):
            S = trial
    
    return S

def construct():
    """Return up to 18 distinct integers in 2..40 with all subset products distinct."""
    t0 = time.time()
    
    S = _build_candidates()
    
    # Ensure list of Python ints
    S = [int(x) for x in S]
    
    # Trim to max 18 if needed
    if len(S) > 18:
        S = S[:18]
    
    # Final verification
    if time.time() - t0 < _DEADLINE - 2:
        prods = [1]
        for a in S:
            prods = prods + [x * a for x in prods]
        if len(prods) != len(set(prods)):
            # Fallback: return just primes
            S = [p for p in range(2, 41) if _is_prime(p)]
    
    return S
# EVOLVE-BLOCK-END
