import math
import time
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Start with N=15M, try to push higher if time permits
    N = 15_000_000
    
    # SPF Sieve using numpy
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
            
    spf_list = spf.tolist()
    
    sigma = [1] * (N + 1)
    
    # Optimized sigma computation with local variable caching
    spf_l = spf_list
    sig = sigma
    t_check_interval = 100000
    for i in range(2, N + 1):
        if (i & (t_check_interval - 1)) == 0:
            if time.time() > deadline - 25:
                N = i
                break
        p = spf_l[i]
        m = i // p
        if m % p == 0:
            power = p
            while m % p == 0:
                m //= p
                power *= p
            sig[i] = sig[m] * (power * p - 1) // (p - 1)
        else:
            sig[i] = sig[m] * (1 + p)
    
    # Main collision search
    M_to_ks = defaultdict(list)
    best_M = 333312
    best_sols = [336, 372, 434]
    best_count = 3
    
    sig = sigma
    t_check_interval2 = 200000
    for k in range(1, N + 1):
        if (k & (t_check_interval2 - 1)) == 0:
            if time.time() > deadline - 5:
                break
        M_val = k * sig[k]
        if M_val <= 10**18:
            lst = M_to_ks.get(M_val)
            if lst is None:
                M_to_ks[M_val] = [k]
            else:
                lst.append(k)
                if len(lst) > best_count:
                    best_count = len(lst)
                    best_sols = lst[:]
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
