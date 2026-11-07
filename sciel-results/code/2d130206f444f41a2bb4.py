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

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin

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

    # Try to increase N
    # Estimate time for N=10^7
    # Slicing loop: ~10^7 iterations.
    # Vectorized operations: ~10^7 * log(10^7) ~ 1.6 * 10^8 ops.
    
    N = 10_000_000
    
    try:
        sigma = np.zeros(N + 1, dtype=np.int64)
        for d in range(1, N + 1):
            sigma[d::d] += d
            if time.time() > deadline - 40.0:
                break
    except Exception:
        return _fallback()

    if time.time() > deadline - 30.0:
        return _fallback()

    # Vectorized computation of M = k * sigma(k)
    ks = np.arange(1, N + 1, dtype=np.int64)
    sigmas = sigma[1:] # sigma[1], ..., sigma[N]
    
    # Check for overflow? k * sigma(k) can exceed int64 max (9.2e18) if M > 9e18.
    # But we filter M <= 1e18.
    # However, intermediate product might overflow int64.
    # 1e7 * 1e7 * 1.6 = 1.6e14. Safe.
    # Max k is 1e7. Max sigma(k) approx 1.6e7. Product approx 1.6e14.
    # So int64 is safe for N=1e7.
    
    M_vals = ks * sigmas
    
    # Filter M <= 10^18
    valid_mask = M_vals <= 10**18
    if not np.any(valid_mask):
        return _fallback()
        
    M_valid = M_vals[valid_mask]
    k_valid = ks[valid_mask]
    
    # Find M with most occurrences
    # unique_Ms, inverse = np.unique(M_valid, return_inverse=True)
    # This sorts and is O(N log N).
    # N_valid is roughly N.
    
    if time.time() > deadline - 20.0:
        return _fallback()
        
    unique_Ms, inverse_indices, counts = np.unique(M_valid, return_inverse=True, return_counts=True)
    
    if len(unique_Ms) == 0:
        return _fallback()
        
    max_count = np.max(counts)
    if max_count < 2:
        # No collisions found? Unlikely for large N.
        # But if no collision, we return the best single one?
        # Task requires distinct k_i. Score is number of solutions.
        # If max_count is 1, we can still return 1 solution?
        # "score is the number of listed solutions".
        # So 1 is better than 0.
        # But we want to maximize.
        pass
        
    best_idx = np.argmax(counts)
    best_M = int(unique_Ms[best_idx])
    
    # Get k's for best_M
    # We need to find indices in k_valid where M_valid == best_M
    # Or use inverse_indices
    # inverse_indices maps each element in M_valid to its index in unique_Ms
    # So we want indices i such that inverse_indices[i] == best_idx
    
    # This is another O(N) pass or search.
    # Since we already have inverse_indices, we can use it.
    # But we need to map back to k_valid.
    
    # k_valid is aligned with M_valid.
    # So we just need indices in k_valid where M_valid is best_M.
    # Using inverse_indices:
    target_indices = np.where(inverse_indices == best_idx)[0]
    
    if len(target_indices) == 0:
        return _fallback()
        
    candidate_ks = k_valid[target_indices]
    
    # Filter k <= sqrt(M)
    limit = math.isqrt(best_M)
    valid_ks = [int(k) for k in candidate_ks if k <= limit]
    
    if not valid_ks:
        return _fallback()
        
    # Final verification
    verified = []
    for k in valid_ks:
        if 1 <= k <= limit and k * _sigma_slow(k) == best_M:
            verified.append(k)
            
    if not verified:
        return _fallback()
        
    verified = sorted(set(verified))
    return {'M': int(best_M), 'k': verified}
