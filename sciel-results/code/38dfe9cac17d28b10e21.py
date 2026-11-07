import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    LIMIT = 10**18
    
    # Choose N based on available time and memory
    # N = 10**7 is safe. Let's try 2 * 10**7
    N = 2 * 10**7
    
    # Linear Sieve for Sigma
    sigma = [0] * (N + 1)
    sigma[1] = 1
    primes = []
    is_composite = [False] * (N + 1)
    spf = [0] * (N + 1)       # Smallest Prime Factor
    spow = [1] * (N + 1)      # Highest power of spf dividing i
    sigma_spf = [1] * (N + 1) # Sum of powers of spf: 1 + p + ... + p^a
    
    for i in range(2, N + 1):
        if time.time() > deadline:
            break
        if not is_composite[i]:
            primes.append(i)
            spf[i] = i
            spow[i] = i
            sigma_spf[i] = 1 + i
            sigma[i] = 1 + i
        
        for p in primes:
            if i * p > N:
                break
            is_composite[i * p] = True
            spf[i * p] = p
            if p == spf[i]:
                spow[i * p] = spow[i] * p
                sigma_spf[i * p] = sigma_spf[i] + spow[i * p]
                sigma[i * p] = sigma[i] * sigma_spf[i * p] // sigma_spf[i]
                break
            else:
                spow[i * p] = p
                sigma_spf[i * p] = 1 + p
                sigma[i * p] = sigma[i] * sigma_spf[i * p]
        
        if time.time() > deadline:
            break

    # Find best M
    solutions = defaultdict(list)
    best_M = 1
    best_count = 0
    best_ks = []
    
    for k in range(1, N + 1):
        if time.time() > deadline:
            break
        s = sigma[k]
        if s == 0: 
            continue
        M = k * s
        if M > LIMIT:
            continue
        solutions[M].append(k)
        if len(solutions[M]) > best_count:
            best_count = len(solutions[M])
            best_M = M
            best_ks = solutions[M][:]
            
    if best_count > 0:
        return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
    else:
        return {'M': 333312, 'k': [336, 372, 434]}
