import time
import math
from collections import defaultdict

def _sigma_from_factors(factors):
    """Compute sigma(n) from prime factorization dict {p: e}."""
    if not factors:
        return 1
    result = 1
    for p, e in factors.items():
        # (p^(e+1) - 1) // (p - 1)
        pe = pow(p, e + 1)
        result *= (pe - 1) // (p - 1)
    return result

def _factorize(n):
    """Trial division factorization. Returns dict {p: e}."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def construct():
    deadline = time.time() + 140  # safety margin
    
    # Strategy: enumerate k up to some bound, compute f(k) = k * sigma(k),
    # count preimages, find M with most solutions where all k_i <= isqrt(M).
    
    # First, let's try a moderate enumeration to find good M values.
    # We need k_i <= isqrt(M), so M = k_i * sigma(k_i) >= k_i^2, which is always true.
    # Also k_i * sigma(k_i) = M means sigma(k_i) = M / k_i, so M/k_i must be integer
    # and sigma(k_i) = M/k_i.
    
    # Key insight: for k * sigma(k) = M, we need sigma(k) = M/k.
    # So k must divide M, and sigma(k) = M/k.
    
    # Let's enumerate k up to a bound and group by M = k*sigma(k).
    # For each M, collect all k such that k*sigma(k) = M and k <= isqrt(M).
    
    # Start with a reasonable bound. The seed has M=333312 with 3 solutions.
    # Let's try to find M with more solutions.
    
    # Enumerate k up to max_k and build a dict: M -> list of k
    max_k = 500000  # start here, can increase if time allows
    
    m_to_ks = defaultdict(list)
    
    for k in range(1, max_k + 1):
        if time.time() > deadline:
            break
        # Compute sigma(k)
        # Fast sigma computation via trial division
        n = k
        sig = 1
        d = 2
        tmp = n
        while d * d <= tmp:
            if tmp % d == 0:
                pe = 1
                pk = 1
                while tmp % d == 0:
                    tmp //= d
                    pk *= d
                    pe += pk
                sig *= pe
            d += 1 if d == 2 else 2
        if tmp > 1:
            sig *= (1 + tmp)
        
        M = k * sig
        if M <= 10**18:
            m_to_ks[M].append(k)
    
    # Find M with most valid k (k <= isqrt(M))
    best_M = None
    best_ks = []
    
    for M, ks in m_to_ks.items():
        if time.time() > deadline:
            break
        isqrt_M = math.isqrt(M)
        valid_ks = [k for k in ks if k <= isqrt_M]
        if len(valid_ks) > len(best_ks):
            best_M = M
            best_ks = valid_ks
            if len(best_ks) >= 100:  # diminishing returns, stop early
                break
    
    if best_M is None or not best_ks:
        # Fallback to seed
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Verify with exact factorization
    verified_ks = []
    for k in best_ks:
        if time.time() > deadline:
            break
        factors = _factorize(k)
        sig = _sigma_from_factors(factors)
        if k * sig == best_M and k <= math.isqrt(best_M):
            verified_ks.append(k)
    
    # Ensure distinct and sorted
    verified_ks = sorted(set(verified_ks))
    
    if len(verified_ks) < len(best_ks):
        # Some didn't verify, use what we have
        pass
    
    if not verified_ks:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Cap at 1000
    verified_ks = verified_ks[:1000]
    
    return {'M': int(best_M), 'k': [int(k) for k in verified_ks]}
