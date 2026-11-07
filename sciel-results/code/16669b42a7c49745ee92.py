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
    deadline = start + 110.0

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

    N = 30_000_000

    try:
        sigma = np.zeros(N + 1, dtype=np.int64)
        for d in range(1, N + 1):
            sigma[d::d] += d
            if time.time() > deadline - 60.0:
                break
    except Exception:
        return _fallback()

    if time.time() > deadline - 50.0:
        return _fallback()

    ks = np.arange(1, N + 1, dtype=np.int64)
    M_vals = ks * sigma
    
    # Filter M <= 10^18
    # All M_vals are positive.
    
    # Find duplicates
    unique_M, counts, inverse = np.unique(M_vals, return_counts=True, return_inverse=True)
    
    if time.time() > deadline - 20.0:
        return _fallback()
    
    max_count = np.max(counts)
    if max_count < 2:
        return _fallback()
        
    best_idx = np.argmax(counts)
    best_M = unique_M[best_idx]
    
    mask = M_vals == best_M
    candidate_ks = np.nonzero(mask)[0] + 1
    
    limit = math.isqrt(best_M)
    valid_ks = candidate_ks[candidate_ks <= limit]
    
    if len(valid_ks) == 0:
        return _fallback()
        
    valid_ks_list = [int(x) for x in valid_ks]
    
    verified = []
    for k in valid_ks_list:
        if k * sigma[k] == best_M:
            verified.append(k)
            
    if not verified:
        return _fallback()
        
    return {'M': int(best_M), 'k': verified}
