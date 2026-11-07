"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

DEADLINE = 85.0  # safety margin before 90s kill

# Precompute prime factorizations for 2..256
def _sieve_smallest_prime():
    spf = list(range(257))
    for i in range(2, 257):
        if spf[i] == i:
            for j in range(i * i, 257, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

_spf = _sieve_smallest_prime()

def _factorize(n):
    """Return dict of prime -> exponent for n."""
    factors = {}
    while n > 1:
        p = _spf[n]
        e = 0
        while n % p == 0:
            n //= p
            e += 1
        factors[p] = e
    return factors

# Precompute factorization for all numbers 2..256
_fact = [None] * 257
for i in range(2, 257):
    _fact[i] = _factorize(i)

# Precompute: for each a in 2..256, and for each b in 2..256,
# the set of c in 2..256 such that a | b*c
# This is expensive (255^3), so instead compute on the fly but fast.

# Better: precompute for each a, the "required" prime exponents.
# a | b*c iff for all p^e in a: v_p(b) + v_p(c) >= e

# For fast checking: given a, b, c, check a | b*c
def _divides_product(a, b, c):
    """Check if a divides b*c using precomputed factorizations."""
    fa = _fact[a]
    fb = _fact[b]
    fc = _fact[c]
    for p, e in fa.items():
        vp_b = fb.get(p, 0)
        vp_c = fc.get(p, 0)
        if vp_b + vp_c < e:
            return False
    return True

def _can_add_to_set(S_set, S_list, candidate):
    """Check if adding candidate to S keeps the set valid.
    S_set is a set for O(1) membership, S_list is the list.
    """
    n = len(S_list)
    
    # Case 1: candidate ∤ b*c for all distinct b,c in S
    for i in range(n):
        b = S_list[i]
        for j in range(n):
            if i == j:
                continue
            c = S_list[j]
            if _divides_product(candidate, b, c):
                return False
    
    # Case 2 & 3: for a in S, check a ∤ b*c where one of b,c is candidate
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if j == i:
                continue
            b = S_list[j]
            # Check a | b * candidate
            if _divides_product(a, b, candidate):
                return False
    
    return True

def _is_valid_set(S):
    """Verify the entire set is valid."""
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(n):
            if j == i:
                continue
            b = S[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                c = S[k]
                if _divides_product(a, b, c):
                    return False
    return True

def _greedy_order(order_key, candidates):
    """Greedy construction with given ordering."""
    S = []
    S_set = set()
    for cand in candidates:
        if _can_add_to_set(S_set, S, cand):
            S.append(cand)
            S_set.add(cand)
    return S

def _local_search(S, remaining, deadline):
    """Try to add more elements from remaining to S."""
    S_list = list(S)
    S_set = set(S_list)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for cand in remaining:
            if cand in S_set:
                continue
            if _can_add_to_set(S_set, S_list, cand):
                S_list.append(cand)
                S_set.add(cand)
                improved = True
                break
    return S_list

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    all_candidates = list(range(2, 257))
    best_set = []
    best_size = 0
    
    # Define multiple ordering strategies
    def key_largest_prime_factor(x):
        # Return the largest prime factor
        max_p = 1
        for p in _fact[x]:
            if p > max_p:
                max_p = p
        return max_p
    
    def key_num_prime_factors(x):
        return len(_fact[x])
    
    def key_descending(x):
        return -x
    
    def key_ascending(x):
        return x
    
    def key_prime_first(x):
        # Primes first (by size), then composites
        is_prime = len(_fact[x]) == 1
        return (0 if is_prime else 1, x if is_prime else -x)
    
    def key_lpf_then_size(x):
        return (-key_largest_prime_factor(x), x)
    
    orderings = [
        key_largest_prime_factor,
        key_num_prime_factors,
        key_descending,
        key_ascending,
        key_prime_first,
        key_lpf_then_size,
    ]
    
    # Try each ordering
    for order_key in orderings:
        if time.time() > deadline:
            break
        sorted_cands = sorted(all_candidates, key=order_key)
        S = _greedy_order(order_key, sorted_cands)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
        
        # Local search
        remaining = [x for x in all_candidates if x not in set(S)]
        S = _local_search(S, remaining, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    # Try random orderings with remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 1.0:
        num_random = min(100, int(remaining_time * 5))
        for _ in range(num_random):
            if time.time() > deadline:
                break
            random.shuffle(all_candidates)
            S = _greedy_order(None, all_candidates)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
            
            remaining = [x for x in all_candidates if x not in set(S)]
            S = _local_search(S, remaining, deadline)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
    
    # Final verification
    if best_set and not _is_valid_set(best_set):
        # Fallback: return primes > 128 (always valid)
        best_set = [x for x in range(129, 257) if len(_fact[x]) == 1]
    
    return list(best_set)
# EVOLVE-BLOCK-END
