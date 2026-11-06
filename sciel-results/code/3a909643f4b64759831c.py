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
from itertools import product

def _tau_perp(n):
    """Count adjacent coprime pairs in sorted divisor list of n."""
    if n < 2:
        return 0
    # Generate all divisors
    divs = [1]
    d = 2
    while d * d <= n:
        if n % d == 0:
            for existing in list(divs):
                divs.append(existing * d)
                if d * d != n:
                    divs.append(existing * (n // d))
        d += 1
    if n > 1:
        divs.append(n)
    divs = sorted(set(divs))
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

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
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    # Base: first 15 primes
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    P15 = 1
    for p in base_primes:
        P15 *= p
    
    best_n = P15
    best_score = _tau_perp_fast(P15)
    
    # Strategy: try replacing some large primes with higher exponents of small primes
    # Key insight: 2^a * 3^b * 5^c * ... can have more divisors and different adjacency structure
    
    # Search over exponent vectors for first k primes where k <= 15
    # Focus on configurations that fit in 10^18
    
    # Precompute: for each number of distinct primes k (10 to 15),
    # search exponent allocations on the first k primes
    
    primes_all = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    
    # For each k from 12 to 15, search exponent combinations
    # Use a greedy/beam approach: start with all 1s, then increment exponents of small primes
    # if it fits and improves score
    
    for k in range(12, 16):
        if time.time() > deadline - 5:
            break
        pk = primes_all[:k]
        # Start with all exponents = 1
        exps = [1] * k
        base_val = 1
        for p, e in zip(pk, exps):
            base_val *= (p ** e)
        
        if base_val > 10**18:
            continue
        
        # Try to improve by incrementing exponents of smaller primes
        # Priority: try incrementing exponent of prime[i] for i=0,1,2,...
        # Keep track of current best for this k
        
        current_exps = exps[:]
        current_val = base_val
        current_score = best_score  # will recompute below if needed
        
        # Beam search: maintain top candidates
        candidates = [(current_exps[:], current_val)]
        
        for step in range(200):
            if time.time() > deadline - 3:
                break
            
            new_candidates = []
            for exps_c, val_c in candidates:
                # Try incrementing each prime's exponent
                for i in range(k):
                    new_exp = exps_c[i] + 1
                    new_val = val_c * pk[i]
                    if new_val > 10**18:
                        continue
                    new_exps = exps_c[:]
                    new_exps[i] = new_exp
                    new_candidates.append((new_exps, new_val))
            
            # Also try decrementing and incrementing another (swap)
            # For now, just keep the top candidates by diversity
            
            # Deduplicate and limit
            seen = set()
            unique = []
            for exps_c, val_c in new_candidates:
                key = tuple(exps_c)
                if key not in seen:
                    seen.add(key)
                    unique.append((exps_c, val_c))
            
            # Evaluate scores and keep top by score
            scored = []
            for exps_c, val_c in unique[:50]:  # limit evaluations
                if time.time() > deadline - 2:
                    break
                n_c = 1
                for p, e in zip(pk, exps_c):
                    n_c *= (p ** e)
                s = _tau_perp_fast(n_c)
                scored.append((s, n_c, exps_c, val_c))
            
            scored.sort(key=lambda x: -x[0])
            
            for s, n_c, exps_c, val_c in scored:
                if s > best_score:
                    best_score = s
                    best_n = n_c
            
            # Keep top candidates for next round (diversity + score)
            if scored:
                # Keep top 5 by score + some random
                candidates = [(x[2], x[3]) for x in scored[:5]]
            else:
                break
    
    # Also try specific known-good patterns:
    # 2^a * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
    # with a > 1
    if time.time() < deadline - 5:
        for a in range(2, 8):
            n_try = (2 ** a) * 1
            for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
                n_try *= p
            if n_try <= 10**18:
                s = _tau_perp_fast(n_try)
                if s > best_score:
                    best_score = s
                    best_n = n_try
    
    # Try: 2^a * 3^b * product of remaining primes
    if time.time() < deadline - 5:
        rest_primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        rest_prod = 1
        for p in rest_primes:
            rest_prod *= p
        
        for a in range(1, 10):
            for b in range(1, 5):
                n_try = (2 ** a) * (3 ** b) * rest_prod
                if n_try <= 10**18:
                    s = _tau_perp_fast(n_try)
                    if s > best_score:
                        best_score = s
                        best_n = n_try
    
    # Try: 2^a * 3^b * 5^c * product of remaining
    if time.time() < deadline - 5:
        rest_primes = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        rest_prod = 1
        for p in rest_primes:
            rest_prod *= p
        
        for a in range(1, 8):
            for b in range(1, 4):
                for c in range(1, 3):
                    n_try = (2 ** a) * (3 ** b) * (5 ** c) * rest_prod
                    if n_try <= 10**18:
                        s = _tau_perp_fast(n_try)
                        if s > best_score:
                            best_score = s
                            best_n = n_try
    
    # Try dropping one large prime and increasing exponents
    if time.time() < deadline - 5:
        # Drop 47, use budget for higher exponents
        primes_no47 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
        prod_no47 = 1
        for p in primes_no47:
            prod_no47 *= p
        
        # Now we have budget: 10^18 / prod_no47 for extra exponents
        budget = 10**18 // prod_no47
        
        # Try 2^a * 3^b * 5^c * 7^d * prod_no47 (with base exponents already included)
        # Actually prod_no47 has each prime once. Extra budget for additional powers.
        for a in range(1, min(8, budget // 2 + 1)):
            rem = budget // (2 ** a)
            if rem < 1:
                break
            for b in range(1, min(4, rem // 3 + 1)):
                rem2 = rem // (3 ** b)
                if rem2 < 1:
                    break
                for c in range(1, min(3, rem2 // 5 + 1)):
                    rem3 = rem2 // (5 ** c)
                    if rem3 < 1:
                        break
                    n_try = prod_no47 * (2 ** a) * (3 ** b) * (5 ** c)
                    # Wait, prod_no47 already includes 2,3,5 once each
                    # So n_try = 2^(1+a) * 3^(1+b) * 5^(1+c) * 7 * ... * 43
                    # That's wrong. Let me recalculate.
                    # prod_no47 = 2*3*5*7*...*43
                    # If I want 2^(1+a), I need to multiply by 2^a additional
                    # So n_try = prod_no47 * 2^a * 3^b * 5^c
                    if n_try <= 10**18:
                        s = _tau_perp_fast(n_try)
                        if s > best_score:
                            best_score = s
                            best_n = n_try
    
    # Try with 14 primes (drop 43 and 47)
    if time.time() < deadline - 5:
        primes_14 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
        prod_14 = 1
        for p in primes_14:
            prod_14 *= p
        
        budget = 10**18 // prod_14
        # Try various exponent increases
        for a in range(1, min(10, budget // 2 + 1)):
            rem = budget // (2 ** a)
            if rem < 1:
                break
            for b in range(1, min(6, rem // 3 + 1)):
                rem2 = rem // (3 ** b)
                if rem2 < 1:
                    break
                for c in range(1, min(4, rem2 // 5 + 1)):
                    rem3 = rem2 // (5 ** c)
                    if rem3 < 1:
                        break
                    n_try = prod_14 * (2 ** a) * (3 ** b) * (5 ** c)
                    if n_try <= 10**18:
                        s = _tau_perp_fast(n_try)
                        if s > best_score:
                            best_score = s
                            best_n = n_try
    
    # Try with 13 primes (drop 41, 43, 47)
    if time.time() < deadline - 5:
        primes_13 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
        prod_13 = 1
        for p in primes_13:
            prod_13 *= p
        
        budget = 10**18 // prod_13
        for a in range(1, min(12, budget // 2 + 1)):
            rem = budget // (2 ** a)
            if rem < 1:
                break
            for b in range(1, min(7, rem // 3 + 1)):
                rem2 = rem // (3 ** b)
                if rem2 < 1:
                    break
                for c in range(1, min(5, rem2 // 5 + 1)):
                    rem3 = rem2 // (5 ** c)
                    if rem3 < 1:
                        break
                    for d in range(1, min(3, rem3 // 7 + 1)):
                        n_try = prod_13 * (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                        if n_try <= 10**18:
                            s = _tau_perp_fast(n_try)
                            if s > best_score:
                                best_score = s
                                best_n = n_try
    
    return best_n
# EVOLVE-BLOCK-END
