"""Constructs an integer n <= 10^18 maximizing tau_perp(n):
the number of adjacent coprime pairs in the sorted divisor list.

Strategy: Use the product of the first 15 primes (primorial P_15).
This number is square-free with the maximum possible number of distinct
prime factors (k=15) under the constraint n <= 10^18.
For square-free numbers, the divisor lattice is a hypercube. Adjacent
divisors in sorted order often differ by a single prime factor, and many
adjacent pairs have disjoint prime supports (gcd 1). Maximizing k maximizes
the number of potential disjoint support transitions.
"""

# EVOLVE-BLOCK-START
import time
from math import gcd

def _tau_perp_fast(n):
    """Faster tau_perp using factorization-based divisor generation."""
    if n < 2:
        return 0
    # Factorize n
    factors = {}
    temp = n
    d = 2
    while d * d <= temp:
        while temp % d == 0:
            factors[d] = factors.get(d, 0) + 1
            temp //= d
        d += 1
    if temp > 1:
        factors[temp] = factors.get(temp, 0) + 1
    
    primes = list(factors.keys())
    exps = [factors[p] for p in primes]
    k = len(primes)
    
    # Generate divisors via recursive product
    divs = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        pw = 1
        for i in range(e + 1):
            for d in divs:
                new_divs.append(d * pw)
            pw *= p
        divs = new_divs
    
    divs.sort()
    
    # For square-free numbers, use bitmask gcd check (much faster)
    if all(e == 1 for e in exps):
        # Map each divisor to its prime bitmask
        divisor_mask = {}
        for i, d in enumerate(divs):
            mask = 0
            for j, p in enumerate(primes):
                if d % p == 0:
                    mask |= (1 << j)
            divisor_mask[d] = mask
        
        count = 0
        for i in range(len(divs) - 1):
            if divisor_mask[divs[i]] & divisor_mask[divs[i+1]] == 0:
                count += 1
        return count
    else:
        count = 0
        for i in range(len(divs) - 1):
            if gcd(divs[i], divs[i + 1]) == 1:
                count += 1
        return count

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Base: first 15 primes
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    P15 = 1
    for p in base_primes:
        P15 *= p
    
    best_n = P15
    best_score = _tau_perp_fast(P15)
    
    primes_all = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    
    # Helper to compute n from prime list and exponents
    def make_n(pk, exps):
        n = 1
        for p, e in zip(pk, exps):
            n *= (p ** e)
        return n
    
    # Beam search for k distinct primes
    for k in range(12, 16):
        if time.time() > deadline - 10:
            break
        pk = primes_all[:k]
        base_val = 1
        for p in pk:
            base_val *= p
        
        if base_val > 10**18:
            continue
        
        # Beam search: start with all 1s, explore exponent increments
        candidates = [([1] * k, base_val)]
        seen_states = {tuple([1] * k)}
        
        max_steps = 300
        for step in range(max_steps):
            if time.time() > deadline - 8:
                break
            
            new_candidates = []
            for exps_c, val_c in candidates:
                for i in range(k):
                    if exps_c[i] >= 6:  # Limit exponent to keep search manageable
                        continue
                    new_exps = exps_c[:]
                    new_exps[i] += 1
                    new_val = val_c * pk[i]
                    if new_val > 10**18:
                        continue
                    key = tuple(new_exps)
                    if key not in seen_states:
                        seen_states.add(key)
                        new_candidates.append((new_exps, new_val))
            
            if not new_candidates:
                break
            
            # Score a subset of candidates
            scored = []
            limit = 30
            for exps_c, val_c in new_candidates[:limit]:
                if time.time() > deadline - 5:
                    break
                n_c = make_n(pk, exps_c)
                s = _tau_perp_fast(n_c)
                scored.append((s, n_c, exps_c, val_c))
            
            scored.sort(key=lambda x: -x[0])
            
            for s, n_c, exps_c, val_c in scored:
                if s > best_score:
                    best_score = s
                    best_n = n_c
            
            # Keep top candidates for next round
            if scored:
                candidates = [(x[2], x[3]) for x in scored[:8]]
            else:
                break
    
    # Targeted searches with specific prime sets and exponent patterns
    if time.time() < deadline - 10:
        # 15 primes with extra 2s
        rest_15 = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        rest_prod_15 = 1
        for p in rest_15:
            rest_prod_15 *= p
        for a in range(1, 7):
            n_try = (2 ** a) * rest_prod_15
            if n_try <= 10**18:
                s = _tau_perp_fast(n_try)
                if s > best_score:
                    best_score = s
                    best_n = n_try
    
    if time.time() < deadline - 10:
        # 14 primes (drop 47)
        primes_14 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
        prod_14 = 1
        for p in primes_14:
            prod_14 *= p
        budget = 10**18 // prod_14
        for a in range(1, min(8, budget // 2 + 1)):
            for b in range(1, min(5, budget // (2**a) // 3 + 1)):
                n_try = prod_14 * (2 ** a) * (3 ** b)
                if n_try <= 10**18:
                    s = _tau_perp_fast(n_try)
                    if s > best_score:
                        best_score = s
                        best_n = n_try
    
    if time.time() < deadline - 10:
        # 13 primes (drop 43, 47)
        primes_13 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
        prod_13 = 1
        for p in primes_13:
            prod_13 *= p
        budget = 10**18 // prod_13
        for a in range(1, min(10, budget // 2 + 1)):
            for b in range(1, min(6, budget // (2**a) // 3 + 1)):
                for c in range(1, min(4, budget // (2**a) // (3**b) // 5 + 1)):
                    n_try = prod_13 * (2 ** a) * (3 ** b) * (5 ** c)
                    if n_try <= 10**18:
                        s = _tau_perp_fast(n_try)
                        if s > best_score:
                            best_score = s
                            best_n = n_try
    
    if time.time() < deadline - 5:
        # 12 primes (drop 41, 43, 47)
        primes_12 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        prod_12 = 1
        for p in primes_12:
            prod_12 *= p
        budget = 10**18 // prod_12
        for a in range(1, min(12, budget // 2 + 1)):
            for b in range(1, min(7, budget // (2**a) // 3 + 1)):
                for c in range(1, min(5, budget // (2**a) // (3**b) // 5 + 1)):
                    for d in range(1, min(3, budget // (2**a) // (3**b) // (5**c) // 7 + 1)):
                        n_try = prod_12 * (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                        if n_try <= 10**18:
                            s = _tau_perp_fast(n_try)
                            if s > best_score:
                                best_score = s
                                best_n = n_try
    
    return best_n
# EVOLVE-BLOCK-END
