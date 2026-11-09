# EVOLVE-BLOCK-START
import math
import time
from math import isqrt
import numpy as np

# Precompute primes up to a limit to speed up sigma calculation
_PRIME_LIMIT = 100000
_primes = []
_is_prime = [True] * (_PRIME_LIMIT + 1)
_is_prime[0] = False
_is_prime[1] = False
for i in range(2, int(_PRIME_LIMIT**0.5) + 1):
    if _is_prime[i]:
        _is_prime[i*i::_PRIME_LIMIT+1] = [False] * len(_is_prime[i*i::_PRIME_LIMIT+1])
for i in range(2, _PRIME_LIMIT + 1):
    if _is_prime[i]:
        _primes.append(i)

def _sigma(n):
    """Compute sigma(n) = sum of divisors of n using precomputed primes."""
    if n <= 0:
        return 0
    result = 1
    temp = n
    for p in _primes:
        if p * p > temp:
            break
        if temp % p == 0:
            p_pow = 1
            while temp % p == 0:
                temp //= p
                p_pow *= p
            result *= (p * p_pow - 1) // (p - 1)
    if temp > 1:
        result *= (temp + 1)
    return result

def _factorize(n):
    """Return list of (prime, exponent) for n."""
    if n <= 1:
        return []
    factors = []
    temp = n
    if temp % 2 == 0:
        cnt = 0
        while temp % 2 == 0:
            temp //= 2
            cnt += 1
        factors.append((2, cnt))
    f = 3
    while f * f <= temp:
        if temp % f == 0:
            cnt = 0
            while temp % f == 0:
                temp //= f
                cnt += 1
            factors.append((f, cnt))
        f += 2
    if temp > 1:
        factors.append((temp, 1))
    return factors

def _all_divisors_from_factors(factors):
    """Generate all divisors from prime factorization."""
    divisors = [1]
    for p, e in factors:
        new_divs = []
        p_pow = 1
        for i in range(e + 1):
            for d in divisors:
                new_divs.append(d * p_pow)
            p_pow *= p
        divisors = new_divs
    return divisors

def _find_k_for_M(M):
    """
    Find all k <= sqrt(M) such that k * sigma(k) = M.
    """
    sqrtM = isqrt(M)
    results = []
    factors = _factorize(M)
    divisors = _all_divisors_from_factors(factors)
    
    for k in divisors:
        if k > sqrtM:
            continue
        if k < 1:
            continue
        s = _sigma(k)
        if k * s == M:
            results.append(k)
    return results

def _search_collisions(max_k, deadline):
    """
    Search for collisions in k*sigma(k) for k up to max_k.
    """
    from collections import defaultdict
    collision_map = defaultdict(list)
    
    for k in range(1, max_k + 1):
        if time.time() > deadline:
            break
        s = _sigma(k)
        M = k * s
        if M <= 10**18:
            collision_map[M].append(k)
    return collision_map

def _compute_sigma_sieve(N):
    """
    Compute sigma(n) for n=1..N using a sieve.
    """
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
    return sigma

def construct():
    start_time = time.time()
    deadline = start_time + 105  # 15s buffer
    
    # Use sieve for faster collision search
    L = 5_000_000
    
    try:
        sigma = _compute_sigma_sieve(L)
    except Exception:
        L = 1_000_000
        try:
            sigma = _compute_sigma_sieve(L)
        except Exception:
            return {'M': 333312, 'k': [336, 372, 434]}
            
    if time.time() > deadline:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    k_vals = np.arange(1, L + 1, dtype=np.int64)
    M_vals = k_vals * sigma[1:L+1]
    
    mask = M_vals <= 10**18
    M_vals = M_vals[mask]
    k_vals = k_vals[mask]
    
    sorted_indices = np.argsort(M_vals)
    sorted_M = M_vals[sorted_indices]
    sorted_k = k_vals[sorted_indices]
    
    best_count = 0
    best_M = 0
    best_ks = []
    
    n = len(sorted_M)
    i = 0
    while i < n:
        if time.time() > deadline:
            break
        j = i + 1
        while j < n and sorted_M[j] == sorted_M[i]:
            j += 1
            
        if j - i > best_count:
            M_val = int(sorted_M[i])
            group_k = sorted_k[i:j]
            
            sqrt_M = isqrt(M_val)
            valid_ks = group_k[group_k <= sqrt_M]
            
            if len(valid_ks) > best_count:
                best_count = len(valid_ks)
                best_M = M_val
                best_ks = valid_ks.tolist()
                
        i = j
        
    if best_count == 0:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    best_ks = [int(k) for k in best_ks]
    
    return {'M': int(best_M), 'k': best_ks}
# EVOLVE-BLOCK-END
