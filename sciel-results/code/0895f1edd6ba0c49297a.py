import math
import time
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

def construct():
    start = time.time()
    deadline = start + 140.0  # Safety margin

    def _fallback():
        # Known small example: M = 333312, k in {336, 372, 434}
        M = 333312
        ks = [336, 372, 434]
        ok = []
        for k in ks:
            if 1 <= k <= math.isqrt(M) and k * _sigma_slow(k) == M:
                ok.append(k)
        if len(ok) >= 1:
            return {'M': int(M), 'k': [int(x) for x in ok]}
        return {'M': 1, 'k': [1]}

    # Try to find a better solution with a larger N
    # N=100,000,000
    # sigma as int32 to save memory
    N = 100_000_000
    
    try:
        # Allocate sigma array (int32 is sufficient for sigma(k) < 2^31 for k < 10^8)
        sigma = np.zeros(N + 1, dtype=np.int32)
        
        # Vectorized sigma sieve
        for d in range(1, N + 1):
            sigma[d::d] += d
            if time.time() > deadline - 30.0:
                break
                
    except Exception:
        return _fallback()

    if time.time() > deadline - 20.0:
        return _fallback()

    try:
        # Compute M_k = k * sigma(k)
        # k is up to N. Use int64 for M to avoid overflow (M can be up to 10^18)
        ks = np.arange(1, N + 1, dtype=np.int64)
        # sigma is int32, convert to int64 for multiplication to be safe, or let numpy handle it
        M_vals = ks * sigma[1:].astype(np.int64)
        
        # Filter M <= 10^18
        mask = M_vals <= 10**18
        M_valid = M_vals[mask]
        k_valid = ks[mask]
        
        if len(M_valid) == 0:
            return _fallback()
        
        # Find collisions
        unique_M, inverse, counts = np.unique(M_valid, return_inverse=True, return_counts=True)
        
        max_count = np.max(counts)
        
        if max_count < 2:
            return _fallback()
            
        # Find the M with the max count
        best_idx = np.argmax(counts)
        best_M = unique_M[best_idx]
        
        # Get all k values that map to best_M
        target_indices = np.where(inverse == best_idx)[0]
        best_ks = k_valid[target_indices]
        
        if len(best_ks) < 2:
            return _fallback()
            
        # Convert to Python list of ints
        best_ks_sorted = sorted([int(k) for k in best_ks])
        
        # Final check: k <= isqrt(M)
        limit = math.isqrt(best_M)
        verified_ks = [k for k in best_ks_sorted if k <= limit]
        
        if not verified_ks:
            return _fallback()
            
        return {'M': int(best_M), 'k': verified_ks}
        
    except Exception:
        return _fallback()
