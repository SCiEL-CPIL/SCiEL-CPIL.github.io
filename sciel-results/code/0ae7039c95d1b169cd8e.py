import time
import random

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

def _build_max_set():
    primes = [p for p in range(2, 41) if _is_prime(p)]
    composites = [c for c in range(2, 41) if not _is_prime(c)]
    
    best_set = list(primes)
    
    # Try different orders of composites to find a larger set
    # We can also try starting with different subsets of primes? 
    # Primes are always safe to include. Removing a prime might allow adding two composites?
    # Unlikely, as primes are the most "independent".
    
    # Strategy: Randomized Greedy
    # 1. Start with primes.
    # 2. Shuffle composites.
    # 3. Add if valid.
    # 4. Repeat many times.
    
    deadline = time.time() + 85
    max_found = len(primes)
    best_candidate = list(primes)
    
    for _ in range(200): # 200 restarts
        if time.time() > deadline:
            break
            
        S = list(primes)
        random.shuffle(composites)
        
        for c in composites:
            if len(S) >= 18:
                break
            trial = S + [c]
            if _has_distinct_subset_products(trial):
                S = trial
                
        if len(S) > max_found:
            max_found = len(S)
            best_candidate = S
            if max_found == 18:
                break
                
    return best_candidate

def construct():
    t0 = time.time()
    S = _build_max_set()
    
    # Ensure list of Python ints
    S = [int(x) for x in S]
    
    # Trim to max 18 if needed
    if len(S) > 18:
        S = S[:18]
        
    # Final verification
    if time.time() - t0 < 88:
        prods = [1]
        for a in S:
            prods = prods + [x * a for x in prods]
        if len(prods) != len(set(prods)):
            # Fallback: return just primes
            S = [p for p in range(2, 41) if _is_prime(p)]
            
    return S
