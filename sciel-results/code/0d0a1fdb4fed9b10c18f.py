"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

_N = 256

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit) if is_prime[i]]

PRIMES = _sieve_primes(_N)

def _prime_factors(n):
    if n == 1:
        return frozenset()
    factors = set()
    for p in PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            factors.add(p)
            while n % p == 0:
                n //= p
    if n > 1:
        factors.add(n)
    return frozenset(factors)

_NUM_FACTORS = [None] + [_prime_factors(i) for i in range(1, _N + 1)]

def construct():
    """
    Find a set of distinct integers from 1..256 such that every pair is coprime,
    maximizing their sum.
    Uses Greedy + Local Search.
    """
    deadline = time.time() + 50  # Safety margin
    
    # 1. Greedy Initialization
    # Sort candidates by value descending. 
    # To break ties or improve greedy, we can prioritize numbers with fewer prime factors 
    # (more "flexible" or "pure") or just value.
    # Let's stick to value descending first.
    candidates = list(range(2, _N + 1))
    candidates.sort(key=lambda x: -x)
    
    used_primes = set()
    selected = {1}
    total = 1
    
    for n in candidates:
        if time.time() > deadline:
            break
        pf = _NUM_FACTORS[n]
        if pf.isdisjoint(used_primes):
            used_primes.update(pf)
            selected.add(n)
            total += n

    # 2. Local Search
    # We try to improve the set by:
    # a) Adding a number that is coprime to all others (if it increases sum, which it always does if positive)
    # b) Swapping a selected number with an unselected number if the swap increases the total sum
    #    and maintains the coprime property.
    
    # Precompute masks for faster disjoint checks
    prime_to_bit = {p: i for i, p in enumerate(PRIMES)}
    masks = {}
    for i in range(1, _N + 1):
        pf = _NUM_FACTORS[i]
        mask = 0
        for p in pf:
            mask |= (1 << prime_to_bit[p])
        masks[i] = mask
        
    used_mask = 0
    for n in selected:
        used_mask |= masks[n]
        
    unselected = set(range(2, _N + 1)) - selected
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add any unselected number that is coprime to all selected
        # Since we want max sum, we check from largest to smallest.
        # If we can add it, we do it.
        # Note: Adding a number might block future larger numbers, but greedy add is a good heuristic.
        # However, swapping is more powerful.
        
        # 1. Try Swaps
        # For each selected number s, try to replace it with an unselected number u
        # such that u > s and u is coprime to (selected - {s}).
        
        selected_list = list(selected)
        # Sort selected by value ascending to try replacing smaller ones first? 
        # Or just iterate.
        
        for s in selected_list:
            if time.time() > deadline:
                break
            # Primes used by other selected numbers
            other_mask = used_mask & ~masks[s]
            
            # Find best u in unselected such that masks[u] & other_mask == 0 and u > s
            best_u = None
            for u in unselected:
                if u > s and (masks[u] & other_mask == 0):
                    # Check if u is coprime to s? No, s is being removed.
                    # But u must be coprime to all OTHERS.
                    # If u > s, sum increases.
                    # We want the largest such u to maximize gain immediately.
                    # But maybe a slightly smaller u allows adding another number later?
                    # Let's just take the largest valid u for now.
                    if best_u is None or u > best_u:
                        best_u = u
            
            if best_u is not None:
                # Perform swap
                selected.remove(s)
                selected.add(best_u)
                unselected.remove(best_u)
                unselected.add(s)
                
                # Update mask
                used_mask = (used_mask & ~masks[s]) | masks[best_u]
                
                total += best_u - s
                improved = True
                # Restart loop as state changed significantly
                break
        
        # 2. Try Adding (if we have "room" in terms of primes, though usually we are packed)
        # If we can add a number without removing anything, do it.
        # This is only possible if there are free primes.
        # With 1..256, we likely use most small primes.
        # But 1 is there.
        # Let's check if any unselected number is coprime to ALL selected.
        # This is equivalent to masks[u] & used_mask == 0.
        
        add_candidate = None
        for u in unselected:
            if masks[u] & used_mask == 0:
                if add_candidate is None or u > add_candidate:
                    add_candidate = u
        
        if add_candidate is not None:
            selected.add(add_candidate)
            unselected.remove(add_candidate)
            used_mask |= masks[add_candidate]
            total += add_candidate
            improved = True
            # Restart loop
            continue

    result = sorted(selected)
    
    # Verification
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # Fallback to just 1 if something went wrong (should not happen)
                return [1]
                
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
