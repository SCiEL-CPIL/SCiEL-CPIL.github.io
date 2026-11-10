import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115s deadline to be safe
    
    # Increased N_MAX to find more collisions
    N_MAX = 4_000_000
    
    # 1. Sieve for smallest prime factors (SPF)
    spf = list(range(N_MAX + 1))
    for i in range(2, int(math.isqrt(N_MAX)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N_MAX + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute sigma(k) using multiplicative property with SPF
    p_power = [1] * (N_MAX + 1)
    rest = [1] * (N_MAX + 1)
    sigma = [0] * (N_MAX + 1)
    sigma[1] = 1
    
    for i in range(2, N_MAX + 1):
        p = spf[i]
        m = i // p
        if m % p == 0:
            p_power[i] = p * p_power[m]
            rest[i] = rest[m]
        else:
            p_power[i] = p
            rest[i] = m
        
        # Geometric sum: 1 + p + ... + p^a where p_power[i] = p^a
        if p == 2:
            geom_sum = 2 * p_power[i] - 1
        else:
            geom_sum = (p * p_power[i] - 1) // (p - 1)
        
        sigma[i] = sigma[rest[i]] * geom_sum
    
    # 3. Find M with maximum collisions
    counts = {}
    
    for k in range(1, N_MAX + 1):
        M = k * sigma[k]
        if M > 10**18:
            continue
        
        if M in counts:
            counts[M] += 1
        else:
            counts[M] = 1
        
        if time.time() > deadline - 5.0:
            break
    
    if not counts:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Find M with max count
    best_M = max(counts, key=counts.get)
    best_count = counts[best_M]
    
    # Collect k's for best_M
    limit = math.isqrt(best_M)
    k_list = []
    
    for k in range(1, min(N_MAX, limit) + 1):
        if k * sigma[k] == best_M:
            k_list.append(k)
    
    k_list = list(set(k_list))
    k_list.sort()
    
    if not k_list:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    return {'M': best_M, 'k': k_list}
