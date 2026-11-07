import math
import time
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit

    # We will search up to this limit for k
    # 25 million is a balance between finding more collisions and memory/time
    N_MAX = 25000000

    # Precompute sigma for all k using a divisor-sum sieve with NumPy
    # Use int64 to be safe for larger k, but int32 is fine for 25M
    sigma_arr = np.zeros(N_MAX + 1, dtype=np.int64)
    
    # Sieve
    # The loop is the bottleneck.
    for d in range(1, N_MAX + 1):
        if d % 5000 == 0:
            if time.time() > deadline - 30:
                break
        sigma_arr[d::d] += d

    # Group by M = k * sigma(k)
    # To save memory, we only store the list of k's for M's that have > 1 solution.
    # We always store the count.
    m_to_count = {}
    m_to_list = {}
    m_to_first_k = {}
    
    best_count = 0
    best_M = 1
    best_ks = [1]

    # Iterate k
    for k in range(1, N_MAX + 1):
        if k % 50000 == 0:
            if time.time() > deadline - 10:
                break

        s = int(sigma_arr[k])
        M = k * s  # Python int

        if M <= 10**18:
            # Update count
            c = m_to_count.get(M, 0) + 1
            m_to_count[M] = c
            
            if c == 1:
                m_to_first_k[M] = k
            else:
                if c == 2:
                    m_to_list[M] = [m_to_first_k[M], k]
                else:
                    m_to_list[M].append(k)
                
                if c > best_count:
                    best_count = c
                    best_M = M
                    best_ks = m_to_list[M]

    # Verify
    def sigma_exact(n):
        if n == 1: return 1
        result = 1
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                p_power = 1
                while temp % p == 0:
                    temp //= p
                    exp += 1
                    p_power *= p
                result *= (p_power * p - 1) // (p - 1)
            p += 1 if p == 2 else 2
        if temp > 1:
            result *= (temp * temp - 1) // (temp - 1)
        return result

    final_ks = []
    for k in best_ks:
        if k * sigma_exact(k) == best_M:
            final_ks.append(k)
            
    if not final_ks:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': int(best_M), 'k': [int(k) for k in final_ks]}
