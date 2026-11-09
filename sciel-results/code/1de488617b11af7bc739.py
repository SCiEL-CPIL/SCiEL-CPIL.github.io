"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

def _sigma(n):
    """Compute sigma(n) - sum of divisors of n."""
    if n <= 1:
        return n
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            exp = 0
            while temp % d == 0:
                temp //= d
                exp += 1
            result *= (d**(exp+1) - 1) // (d - 1)
        d += 1
    if temp > 1:
        result *= (temp + 1)
    return result

def _sigma_sieve(limit):
    """Compute sigma(n) for all n up to limit using sieve."""
    sigma = list(range(limit + 1))
    for d in range(2, limit + 1):
        for m in range(d, limit + 1, d):
            sigma[m] += d
    return sigma

def _sieve_search(deadline, max_k):
    """Sieve-based search: compute k*sigma(k) for k up to max_k, find M with most solutions."""
    start = time.time()
    sigma = _sigma_sieve(max_k)
    
    # Map M -> list of k values
    m_to_ks = defaultdict(list)
    
    for k in range(1, max_k + 1):
        M = k * sigma[k]
        if M <= 10**18:
            m_to_ks[M].append(k)
        if time.time() > deadline:
            break
    
    # Find M with most valid solutions (k <= isqrt(M))
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    for M, ks in m_to_ks.items():
        if len(ks) <= best_count:
            continue
        # Filter: k must satisfy k <= isqrt(M)
        sqrt_M = isqrt(M)
        valid_ks = [k for k in ks if k <= sqrt_M]
        if len(valid_ks) > best_count:
            best_count = len(valid_ks)
            best_M = M
            best_k = valid_ks
    
    return best_M, best_k

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Known good answer
    M0, k0 = 333312, [336, 372, 434]
    best_M, best_k = M0, k0
    
    # Sieve-based search with increasing max_k
    # Start with a reasonable max_k and try to go further if time permits
    for max_k in [200000, 500000, 1000000, 2000000]:
        if time.time() > deadline:
            break
        try:
            found_M, found_k = _sieve_search(deadline, max_k)
            if len(found_k) > len(best_k):
                best_M, best_k = found_M, found_k
        except Exception:
            break
    
    # Final verification
    valid_k = []
    for k in best_k:
        if isinstance(k, int) and not isinstance(k, bool) and 1 <= k <= isqrt(best_M):
            if k * _sigma(k) == best_M:
                valid_k.append(k)
    
    if not valid_k:
        best_M, best_k = M0, k0
        valid_k = [k for k in best_k if k * _sigma(k) == best_M and k <= isqrt(best_M)]
    
    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in valid_k]
    
    return {'M': best_M, 'k': best_k}
# EVOLVE-BLOCK-END
