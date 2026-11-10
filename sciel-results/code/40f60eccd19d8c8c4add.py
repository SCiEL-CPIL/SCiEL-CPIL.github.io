"""Rewritten program: meet-in-the-middle over smooth number factors to find M with many k s.t. k*sigma(k)=M."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

_START_TIME = time.time()
_DEADLINE = 150.0  # evaluator kills at 150s wall-clock; keep safety margin
_SAFETY_MARGIN = 5.0  # seconds to reserve before deadline

def _elapsed():
    return time.time() - _START_TIME

def _deadline_reached():
    return _elapsed() > _DEADLINE - _SAFETY_MARGIN

def _sigma_smooth(n, primes):
    """Compute sigma(n) for n = product of primes^e. Returns (sigma, ok) where ok=False if n too large or prime too big."""
    # n is already factored implicitly by construction in the caller
    # This helper is for direct use
    s = 1
    m = n
    for p in primes:
        if p * p > m:
            break
        if m % p == 0:
            pe = 1
            pe_sum = 1
            while m % p == 0:
                m //= p
                pe *= p
                pe_sum += pe
            s *= pe_sum
    if m > 1:
        s *= (1 + m)
    return s

def _sigma_from_factorization(factors):
    """factors: list of (prime, exponent). Returns sigma."""
    s = 1
    for p, e in factors:
        # sigma(p^e) = (p^(e+1) - 1) / (p - 1)
        pe = 1
        for _ in range(e):
            pe *= p
        pe_sum = (pe * p - 1) // (p - 1)
        s *= pe_sum
    return s

def _generate_smooth(primes, max_val):
    """Generate all numbers of form product of primes^e <= max_val.
    Returns list of (number, list_of_(prime,exponent))."""
    results = [(1, [])]
    for p in primes:
        new_results = []
        for (val, facs) in results:
            pe = 1
            e = 0
            while val * pe <= max_val:
                new_results.append((val * pe, facs + [(p, e)]))
                pe *= p
                e += 1
        results = new_results
    return results

def construct():
    # Strategy: 
    # 1. Generate all 2^a * 3^b numbers up to some bound B1
    # 2. For each, compute k1 = that number, sigma(k1)
    # 3. Generate all 5^c numbers up to some bound B2
    # 4. For each, compute k2 = that number, sigma(k2)
    # 5. For each pair (k1, k2) with k1*k2 <= 10^18 and k1*k2*sigma(k1)*sigma(k2) <= 10^18:
    #    M = k1*sigma(k1)*k2*sigma(k2), and k = k1*k2 is a solution
    # 6. Count solutions per M, find M with max count
    # 
    # But we need k_i <= sqrt(M). Since M = k*sigma(k), and sigma(k) >= k+1 for k>1,
    # M >= k*(k+1), so k <= sqrt(M) roughly. Let's verify: k*sigma(k) = M, sigma(k) >= k+1,
    # so M >= k(k+1) > k^2, hence k < sqrt(M). Good.
    #
    # Actually we need k_i <= isqrt(M). Let's be careful.
    # For k=1: sigma(1)=1, M=1, isqrt(1)=1, k=1<=1. OK.
    # For k>1: sigma(k) >= k+1, so M = k*sigma(k) >= k(k+1). 
    # isqrt(M) >= isqrt(k(k+1)) >= k for k>=1? 
    # k(k+1) = k^2 + k. isqrt(k^2+k) >= k since k^2+k >= k^2. Yes.
    # So k <= isqrt(M) is automatically satisfied.
    
    # Let's use primes [2, 3] for k1 and [5] for k2, or maybe [2,3,5] split differently.
    # Better: use [2,3] for one part and [5,7] for another, or just [2,3] and [5].
    # Let's try [2,3] and [5] first, with bounds that keep M <= 10^18.
    
    # For k1 = 2^a * 3^b, sigma(k1) = (2^(a+1)-1) * (3^(b+1)-1)/2
    # For k2 = 5^c, sigma(k2) = (5^(c+1)-1)/4
    # M = k1 * sigma(k1) * k2 * sigma(k2)
    
    # We want to maximize the number of distinct k = k1*k2 giving the same M.
    # This happens when different (a,b,c) triples give the same product.
    
    # Let's enumerate:
    # a from 0 to ~60 (2^60 ~ 10^18)
    # b from 0 to ~37 (3^37 ~ 10^17)
    # c from 0 to ~26 (5^26 ~ 10^18)
    # But k1*k2 <= 10^18 and M <= 10^18.
    
    # Let's generate all (a,b) with 2^a * 3^b <= 10^9 (so k1 up to 10^9)
    # and all c with 5^c <= 10^9 (so k2 up to 10^9)
    # Then k1*k2 <= 10^18, and we check M <= 10^18.
    
    # Actually, let's be more generous: k1 up to 10^12, k2 up to 10^6, etc.
    # Let's just generate all 2^a*3^b <= 10^18 and all 5^c <= 10^18, but that's too many.
    # 2^a*3^b <= 10^18: a up to 60, b up to 37, but constrained by product. ~ 60*37/2 ~ 1100 values.
    # 5^c <= 10^18: c up to 26. ~26 values.
    # Total pairs: ~1100 * 26 = ~28600. That's manageable.
    
    # Let's generate:
    k1_list = []  # (k1, sigma_k1)
    max_k1 = 10**18
    for a in range(0, 61):
        two_pow = 1 << a
        if two_pow > max_k1:
            break
        for b in range(0, 40):
            k1 = two_pow * (3 ** b)
            if k1 > max_k1:
                break
            # sigma(k1) = (2^(a+1)-1) * (3^(b+1)-1)//2
            sigma_k1 = ((1 << (a + 1)) - 1) * ((3 ** (b + 1) - 1) // 2)
            k1_list.append((k1, sigma_k1))
    
    k2_list = []  # (k2, sigma_k2)
    max_k2 = 10**18
    five_pow = 1
    for c in range(0, 30):
        if five_pow > max_k2:
            break
        k2 = five_pow
        sigma_k2 = (5 ** (c + 1) - 1) // 4
        k2_list.append((k2, sigma_k2))
        five_pow *= 5
    
    # Now for each pair, compute M and k, check constraints
    # M = k1 * sigma_k1 * k2 * sigma_k2
    # k = k1 * k2
    # Need M <= 10^18 and k <= isqrt(M)
    
    solutions = defaultdict(list)  # M -> list of k
    
    for (k1, s1) in k1_list:
        if _deadline_reached():
            break
        for (k2, s2) in k2_list:
            if _deadline_reached():
                break
            k = k1 * k2
            if k > 10**18:
                continue
            M = k1 * s1 * k2 * s2
            if M > 10**18:
                continue
            # Check k <= isqrt(M)
            if k <= math.isqrt(M):
                solutions[M].append(k)
    
    # Find M with most solutions
    best_M = None
    best_k_list = []
    for M, k_list in solutions.items():
        # Deduplicate k_list (shouldn't have duplicates but just in case)
        unique_k = list(set(k_list))
        if len(unique_k) > len(best_k_list):
            best_M = M
            best_k_list = unique_k
    
    if best_M is None:
        # Fallback to known solution
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Verify and trim to at most 1000
    best_k_list = best_k_list[:1000]
    
    # Verify each solution
    verified_k = []
    for k in best_k_list:
        # Compute sigma(k) exactly
        n = k
        s = 1
        m = n
        p = 2
        while p * p <= m:
            if m % p == 0:
                pe_sum = 1
                pe = 1
                while m % p == 0:
                    m //= p
                    pe *= p
                    pe_sum += pe
                s *= pe_sum
            p += 1 if p == 2 else 2
        if m > 1:
            s *= (1 + m)
        if k * s == best_M and k <= math.isqrt(best_M):
            verified_k.append(k)
    
    if not verified_k:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    return {'M': best_M, 'k': verified_k}
# EVOLVE-BLOCK-END
