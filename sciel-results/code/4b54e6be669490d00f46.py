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
    Uses Greedy + Local Search with bitmask optimization.
    """
    deadline = time.time() + 50  # Safety margin before 60s limit
    
    # Precompute bitmask representations for all numbers 1..256
    # Each bit position corresponds to a prime in PRIMES list
    prime_to_bit = {p: i for i, p in enumerate(PRIMES)}
    masks = {}
    for i in range(1, _N + 1):
        pf = _NUM_FACTORS[i]
        mask = 0
        for p in pf:
            if p in prime_to_bit:
                mask |= (1 << prime_to_bit[p])
        masks[i] = mask
    
    # 1. Greedy Initialization
    # Sort candidates by value descending
    candidates = list(range(2, _N + 1))
    candidates.sort(key=lambda x: -x)
    
    used_primes = set()
    selected = {1}
    total = 1
    used_mask = 0
    
    for n in candidates:
        if time.time() > deadline:
            break
        pf = _NUM_FACTORS[n]
        if pf.isdisjoint(used_primes):
            used_primes.update(pf)
            selected.add(n)
            total += n
            used_mask |= masks[n]

    # 2. Local Search
    unselected = set(range(2, _N + 1)) - selected
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try Swaps: For each selected number s, try to replace it with an unselected number u
        # such that u > s and u is coprime to all other selected numbers.
        selected_list = list(selected)
        
        for s in selected_list:
            if time.time() > deadline:
                break
            # Mask of primes used by OTHER selected numbers
            other_mask = used_mask & ~masks[s]
            
            # Find best u in unselected such that masks[u] & other_mask == 0 and u > s
            best_u = None
            best_u_val = 0
            for u in unselected:
                if u > s and (masks[u] & other_mask) == 0:
                    if u > best_u_val:
                        best_u = u
                        best_u_val = u
            
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
        
        # Try Adding: If we can add a number without removing anything
        add_candidate = None
        add_candidate_val = 0
        for u in unselected:
            if (masks[u] & used_mask) == 0:
                if u > add_candidate_val:
                    add_candidate = u
                    add_candidate_val = u
        
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
