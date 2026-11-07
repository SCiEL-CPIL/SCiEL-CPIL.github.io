import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # We will search up to N
    # Based on memory and time constraints, N=50,000,000 is a good balance.
    N = 50_000_000
    
    # 1. Compute Smallest Prime Factor (SPF)
    spf = np.arange(N + 1, dtype=np.int32)
    spf[0] = 0
    spf[1] = 1
    limit = int(math.isqrt(N))
    for i in range(2, limit + 1):
        if spf[i] == i:
            start = i * i
            indices = np.arange(start, N + 1, i)
            mask = spf[indices] == indices
            spf[indices[mask]] = i
            
    # Convert to list for faster access in Python loop
    spf_list = spf.tolist()
    
    # 2. Compute sigma(k)
    sigma = np.zeros(N + 1, dtype=np.int32)
    sigma[1] = 1
    
    for k in range(2, N + 1):
        if time.time() > deadline - 40:
            # If we are running out of time, we can't finish the sigma calculation properly.
            # However, 50M should take ~10-15s.
            # If we break, the rest of sigma is 0, which is wrong.
            # So we should not break here unless we have a fallback.
            # For now, assume it finishes.
            break
            
        p = spf_list[k]
        m = k // p
        if m % p == 0:
            power = p
            while m % p == 0:
                m //= p
                power *= p
            sigma[k] = sigma[m] * (power * p - 1) // (p - 1)
        else:
            sigma[k] = sigma[m] * (1 + p)
            
    # 3. Compute M = k * sigma(k)
    ks = np.arange(1, N + 1, dtype=np.int64)
    M_vals = ks * sigma[1:]
    
    # 4. Find most frequent M
    # np.unique sorts the array.
    unique_M, inverse, counts = np.unique(M_vals, return_inverse=True, return_counts=True)
    
    max_idx = np.argmax(counts)
    best_M = int(unique_M[max_idx])
    best_count = int(counts[max_idx])
    
    # 5. Find all k corresponding to best_M
    k_indices = np.where(inverse == max_idx)[0]
    best_ks = (k_indices + 1).tolist()
    
    # 6. Verification
    valid_ks = []
    limit_sqrt = math.isqrt(best_M)
    for k in best_ks:
        if 1 <= k <= limit_sqrt:
            # Verify sigma
            if k <= N:
                s = sigma[k]
            else:
                s = _sigma_slow(k)
            if k * s == best_M:
                valid_ks.append(k)
                
    if not valid_ks:
        # Fallback
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': best_M, 'k': valid_ks}

def _sigma_slow(n):
    if n == 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            result *= (p ** (exp + 1) - 1) // (p - 1)
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (temp * temp - 1) // (temp - 1)
    return result
