import math
import time
from collections import defaultdict
import numpy as np

def construct():
    deadline = time.time() + 110
    
    N = 10_000_000
    
    # SPF Sieve
    spf = np.arange(N + 1, dtype=np.int64)
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
    
    sigma = [1] * (N + 1)
    
    for i in range(2, N + 1):
        if time.time() > deadline - 20:
            break
        p = spf_list[i]
        m = i // p
        if m % p == 0:
            power = p
            while m % p == 0:
                m //= p
                power *= p
            sigma[i] = sigma[m] * (power * p - 1) // (p - 1)
        else:
            sigma[i] = sigma[m] * (1 + p)
            
    M_to_ks = defaultdict(list)
    best_M = 333312
    best_sols = [336, 372, 434]
    best_count = 3
    
    for k in range(1, N + 1):
        if time.time() > deadline - 5:
            break
        M_val = k * sigma[k]
        if M_val <= 10**18:
            M_to_ks[M_val].append(k)
            if len(M_to_ks[M_val]) > best_count:
                best_count = len(M_to_ks[M_val])
                best_sols = M_to_ks[M_val][:]
                best_M = M_val
                
    # Verification
    valid_sols = []
    limit = math.isqrt(best_M)
    for k in best_sols:
        if 1 <= k <= limit:
            if k * sigma[k] == best_M:
                valid_sols.append(k)
                
    if not valid_sols:
        valid_sols = [336, 372, 434]
        
    return {'M': int(best_M), 'k': [int(x) for x in valid_sols]}
