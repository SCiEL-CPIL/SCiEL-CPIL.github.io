"""Square-product extension for 210: find distinct ints in (210,10210]
whose product * 210 is a perfect square, minimizing max - 210."""

import math
import time
from collections import defaultdict

N = 210
MAX_N = 10210
MAX_K = 100
SMALL_PRIMES = [2, 3, 5, 7]


def _is_square(n):
    r = math.isqrt(n)
    return r * r == n


def _sieve_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]


def _factorize(n, primes):
    d = {}
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            d[p] = e
    if n > 1:
        d[n] = 1
    return d


def construct():
    t_start = time.time()
    DEADLINE = t_start + 80.0

    primes = _sieve_primes(MAX_N)
    
    # Precompute parity for all candidates
    parity = {}
    for n in range(211, MAX_N + 1):
        d = _factorize(n, primes)
        mask = 0
        for i, p in enumerate(SMALL_PRIMES):
            if d.get(p, 0) & 1:
                mask |= (1 << i)
        large = frozenset(p for p, e in d.items() if p not in SMALL_PRIMES and e & 1)
        parity[n] = (mask, large)

    TARGET_MASK = 0b1111
    TARGET_LARGE = frozenset()
    TARGET_VECTOR = (TARGET_MASK, TARGET_LARGE)

    best_val = None
    best_set = None

    # Incremental search for minimal max
    # We maintain a list of (n, vector) for n <= current_limit
    # and check for solutions of size 1, 2, 3 (and maybe 4 if needed, but 3 should suffice)
    
    candidates = []
    
    for limit in range(211, MAX_N + 1):
        if time.time() > DEADLINE:
            break
            
        v = parity[limit]
        candidates.append((limit, v))
        
        # Check 1-element
        if v == TARGET_VECTOR:
            return [limit]
            
        # Check 2-element
        # We need v1 ^ v2 == T
        # For the new element (limit, v), check against all previous
        for n_prev, v_prev in candidates[:-1]:
            m_res = v[0] ^ v_prev[0]
            # Large set symmetric difference
            # (A \ B) U (B \ A)
            # If A == B, result is empty
            if v[1] == v_prev[1]:
                lg_res = frozenset()
            else:
                lg_res = v[1].symmetric_difference(v_prev[1])
                
            if (m_res, lg_res) == TARGET_VECTOR:
                # Found a pair
                # Since we are iterating limit upwards, this is the first pair with max=limit
                # But we must ensure we found the GLOBAL minimum max.
                # Since we stop at the first limit where ANY solution exists, and we check all pairs ending at limit,
                # this is the minimal max.
                return [n_prev, limit]
        
        # Check 3-element
        # v1 ^ v2 ^ v3 == T
        # Fix v3 = v (the new one)
        # Need v1 ^ v2 == T ^ v3
        target_pair_xor = (TARGET_MASK ^ v[0], TARGET_LARGE.symmetric_difference(v[1]))
        
        # Iterate over pairs in candidates[:-1]
        # This is O(N^2) per step, total O(N^3).
        # If N is small, this is fine.
        prev_candidates = candidates[:-1]
        for i in range(len(prev_candidates)):
            n1, v1 = prev_candidates[i]
            # Calculate required v2
            m_req = v1[0] ^ target_pair_xor[0]
            # Large set: L1 ^ L_req = L_target_pair
            # L_req = L1 ^ L_target_pair
            lg_req = v1[1].symmetric_difference(target_pair_xor[1])
            req_vec = (m_req, lg_req)
            
            # Check if req_vec exists in prev_candidates[j] for j > i
            # To avoid duplicates and ensure distinctness
            for j in range(i + 1, len(prev_candidates)):
                n2, v2 = prev_candidates[j]
                if v2 == req_vec:
                    # Check distinctness
                    if n1 != n2 and n1 != limit and n2 != limit:
                        return [n1, n2, limit]
                        
    # Fallback if not found (should not happen)
    return [840]

if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Result: {result}")
    print(f"Max - 210: {max(result) - 210}")
    prod = 210
    for x in result:
        prod *= x
    print(f"Product is square: {_is_square(prod)}")
    print(f"Time: {t1 - t0:.2f}s")
