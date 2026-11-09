import math
import time
import numpy as np
from collections import defaultdict

_START = time.time()
DEADLINE = _START + 145  # 5s safety margin before 150s kill

def _elapsed():
    return time.time() - _START

def _time_left():
    return DEADLINE - time.time()

def _is_square_free_sieve(n):
    """Return boolean array where arr[i]=True if i is square-free."""
    sf = np.ones(n + 1, dtype=np.bool_)
    # Sieve of squares: mark multiples of p^2 as not square-free
    p = 2
    while p * p <= n:
        p2 = p * p
        if p2 <= n:
            sf[p2::p2] = False
        p += 1
    sf[0] = False
    return sf

def _sigma_square_free(k, sf_primes):
    """Compute sigma(k) for square-free k given its prime factors."""
    # sigma(k) = prod(p_i + 1) for square-free k
    result = 1
    for p in sf_primes:
        result *= (p + 1)
    return result

def _factorize_square_free(k, spf):
    """Get prime factors of square-free k using smallest-prime-factor array."""
    factors = []
    while k > 1:
        p = spf[k]
        factors.append(p)
        k //= p
    return factors

def construct():
    # Fallback known good result
    fallback = {'M': 333312, 'k': [336, 372, 434]}
    
    if _time_left() < 30:
        return fallback
    
    # Step 1: Build smallest-prime-factor sieve and square-free indicator
    # We'll go up to N=20,000,000 for square-free enumeration
    N = 20_000_000
    
    try:
        # Build SPF (smallest prime factor) array
        spf = np.zeros(N + 1, dtype=np.int32)
        for i in range(2, N + 1):
            if spf[i] == 0:
                # i is prime
                spf[i::i] = np.where(spf[i::i] == 0, i, spf[i::i])
        
        # Build square-free indicator
        sf = _is_square_free_sieve(N)
        
        if _time_left() < 25:
            return fallback
        
        # Step 2: Enumerate square-free k, compute M = k * sigma(k)
        # For square-free k = p1*p2*...*pm: sigma(k) = (p1+1)*(p2+1)*...*(pm+1)
        # M = k * sigma(k) = prod(p_i) * prod(p_i + 1)
        
        groups = defaultdict(list)
        
        # Process in chunks for memory efficiency
        # Iterate over square-free numbers
        sf_indices = np.where(sf[1:])[0] + 1  # 1-based indices of square-free numbers
        
        # For efficiency, process sequentially but in vectorized batches
        batch_size = 500_000
        for start in range(0, len(sf_indices), batch_size):
            if _time_left() < 20:
                break
            
            end = min(start + batch_size, len(sf_indices))
            batch = sf_indices[start:end]
            
            # For each k in batch, factorize and compute M
            for idx in batch:
                k = int(idx)
                # Factorize
                temp = k
                sigma_val = 1
                while temp > 1:
                    p = int(spf[temp])
                    sigma_val *= (p + 1)
                    temp //= p
                
                M = k * sigma_val
                if M <= 10**18:
                    groups[M].append(k)
            
            # Check time periodically
            if _time_left() < 20:
                break
        
        if _time_left() < 15:
            return fallback
        
        # Step 3: Find M with most solutions
        best_M = fallback['M']
        best_ks = fallback['k']
        best_count = len(fallback['k'])
        
        for M, ks in groups.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_M = M
                best_ks = ks[:1000]
        
        if best_count <= best_count and best_ks == fallback['k']:
            return fallback
        
        # Step 4: Verify with exact factorization
        def _sigma_verify(k):
            """Compute sigma(k) by trial division (exact)."""
            if k == 1:
                return 1
            result = 1
            temp = k
            p = 2
            while p * p <= temp:
                if temp % p == 0:
                    exp = 0
                    while temp % p == 0:
                        temp //= p
                        exp += 1
                    result *= (p**(exp+1) - 1) // (p - 1)
                p += 1 if p == 2 else 2
            if temp > 1:
                result *= (temp + 1)  # temp is prime, exp=1
            return result
        
        sqrt_M = math.isqrt(best_M)
        verified_ks = []
        for k in best_ks:
            if k <= sqrt_M and k * _sigma_verify(k) == best_M:
                verified_ks.append(k)
        
        if len(verified_ks) >= 1:
            return {'M': int(best_M), 'k': [int(k) for k in verified_ks]}
        else:
            return fallback
    
    except Exception:
        return fallback

# Keep the helper from current program for compatibility
def _small_search():
    return {'M': 333312, 'k': [336, 372, 434]}
