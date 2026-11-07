import math
import time
from collections import defaultdict
import numpy as np

def _sigma_slow(n):
    if n == 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            term = 1
            pw = 1
            while temp % p == 0:
                temp //= p
                exp += 1
                pw *= p
                term += pw
            result *= term
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (1 + temp)
    return result

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def _get_divisors(n):
    divisors = [1]
    for p in range(2, int(math.sqrt(n)) + 1):
        if n % p == 0:
            # Factor out p
            while n % p == 0:
                n //= p
            # Add new divisors
            new_divs = []
            for d in divisors:
                pw = p
                while d * pw <= n * p: # This logic is tricky, better to just multiply
                    pass
            # Simpler approach for small n
    # Standard divisor generation
    divs = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(divs)

def construct():
    start = time.time()
    deadline = start + 140.0

    def _fallback():
        candidates = [
            (333312, [336, 372, 434]),
            (1, [1]),
        ]
        for M, ks in candidates:
            ok = []
            for k in ks:
                if 1 <= k <= math.isqrt(M) and k * _sigma_slow(k) == M:
                    ok.append(k)
            if len(ok) >= 1:
                return {'M': int(M), 'k': [int(x) for x in ok]}
        return {'M': 1, 'k': [1]}

    best_M = 1
    best_ks = [1]
    
    # Generate highly composite numbers
    # Primes: 2, 3, 5, 7, 11, 13
    primes = [2, 3, 5, 7, 11, 13]
    
    def gen_hcn(idx, current_val, max_exp, limit):
        if time.time() > deadline - 10.0:
            return
        if idx == len(primes):
            if current_val > 1:
                yield current_val
            return
        p = primes[idx]
        for e in range(max_exp + 1):
            if current_val * (p ** e) > limit:
                break
            yield from gen_hcn(idx + 1, current_val * (p ** e), e, limit)
            
    # Limit for H: We want H * P <= 10^18.
    # P = sigma(k) >= k+1. k <= H.
    # So H * (H+1) <= 10^18 => H ~ 10^9.
    # Let's try H up to 10^10 to be safe, but check product.
    limit_H = 10**10
    
    # To avoid too many H, we can just generate a list of candidate H's
    # HCNs are sparse.
    hcn_candidates = list(gen_hcn(0, 1, 60, limit_H))
    
    # Sort by number of divisors descending? Or just iterate.
    # We want many divisors.
    # Let's filter to keep only those with many divisors.
    # Or just process all. There are ~1000 HCNs up to 10^10?
    
    # Actually, we can just iterate over a range of numbers? No, HCNs are specific.
    # Let's process the generated HCNs.
    
    # Optimization: Precompute sigma for small numbers?
    # No, k can be up to 10^10. _sigma_slow is O(sqrt(k)).
    # sqrt(10^10) = 10^5.
    # If we have 1000 HCNs, and each has ~1000 divisors, that's 10^6 sigma calls.
    # 10^6 * 10^5 ops = 10^11 ops. Too slow.
    
    # We need to be smarter.
    # Only check k if sigma(k) is prime.
    # But we have to compute sigma(k) to know.
    
    # Maybe limit H to 10^8?
    # H ~ 10^8 => sqrt(H) ~ 10^4.
    # 10^6 * 10^4 = 10^10. Still slow in Python.
    
    # Let's reduce the number of HCNs or use a faster sigma.
    # Or, note that we only care about k | H.
    # We can compute sigma(k) for all divisors k of H efficiently?
    # If H = prod p_i^e_i, and k = prod p_i^f_i (f_i <= e_i).
    # sigma(k) = prod (p_i^{f_i+1}-1)/(p_i-1).
    # We can compute this for all divisors in O(num_divisors * num_primes).
    # This is fast!
    
    def sigma_from_factors(factors):
        # factors: list of (p, exp)
        res = 1
        for p, e in factors:
            if e == 0:
                continue
            res *= (p**(e+1) - 1) // (p - 1)
        return res

    def get_prime_factorization(n):
        factors = {}
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors[p] = exp
            p += 1 if p == 2 else 2
        if temp > 1:
            factors[temp] = 1
        return factors

    # Process HCNs
    # To ensure we don't miss better M from non-HCN H, we rely on the fact
    # that HCNs have many divisors, increasing chance of collisions in sigma(k).
    
    for H in hcn_candidates:
        if time.time() > deadline - 5.0:
            break
            
        pf_H = get_prime_factorization(H)
        primes_in_H = sorted(pf_H.keys())
        exps_in_H = [pf_H[p] for p in primes_in_H]
        
        # Generate all divisors and their sigma
        # Recursive generation of divisors
        def gen_divs(idx, current_k, current_sigma):
            if time.time() > deadline - 2.0:
                return
            if idx == len(primes_in_H):
                # Check if current_sigma is prime
                if current_sigma > 1 and _is_prime(current_sigma):
                    P = current_sigma
                    M = H * P
                    if M <= 10**18:
                        # Check k <= sqrt(M)
                        if current_k <= math.isqrt(M):
                            yield current_k, P
                return
                
            p = primes_in_H[idx]
            max_e = exps_in_H[idx]
            
            # sigma(p^0) = 1
            # sigma(p^1) = 1+p
            # ...
            
            # We need to update current_sigma correctly.
            # current_sigma is sigma of the part constructed so far.
            # The new part is p^e.
            # sigma(k) = sigma(prev) * sigma(p^e)
            
            for e in range(max_e + 1):
                if current_k * (p ** e) > H:
                    break
                # Calculate sigma(p^e)
                if e == 0:
                    sigma_pe = 1
                else:
                    sigma_pe = (p**(e+1) - 1) // (p - 1)
                    
                new_k = current_k * (p ** e)
                new_sigma = current_sigma * sigma_pe
                
                yield from gen_divs(idx + 1, new_k, new_sigma)

        # Group by P
        # We can't easily group in a generator without collecting.
        # Let's collect (k, P) pairs.
        pairs = []
        for k, P in gen_divs(0, 1, 1):
            pairs.append((k, P))
            
        # Count occurrences of each P
        p_counts = defaultdict(list)
        for k, P in pairs:
            p_counts[P].append(k)
            
        # Find max count for this H
        for P, ks in p_counts.items():
            if len(ks) > len(best_ks):
                M = H * P
                if M <= 10**18:
                    # Verify all k
                    valid_ks = []
                    for k in ks:
                        if k <= math.isqrt(M) and k * _sigma_slow(k) == M:
                            valid_ks.append(k)
                    if len(valid_ks) > len(best_ks):
                        best_M = M
                        best_ks = valid_ks

    if best_ks:
        return {'M': int(best_M), 'k': [int(x) for x in best_ks]}
    else:
        return _fallback()
