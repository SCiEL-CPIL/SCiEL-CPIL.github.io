import math
import time
from collections import defaultdict

def _sigma(n):
    if n == 1:
        return 1
    result = 1
    temp = n
    # factor out 2
    cnt = 0
    while temp % 2 == 0:
        temp //= 2
        cnt += 1
    if cnt:
        result *= (2**(cnt+1) - 1)
    # odd primes
    p = 3
    while p * p <= temp:
        if temp % p == 0:
            cnt = 0
            while temp % p == 0:
                temp //= p
                cnt += 1
            result *= (p**(cnt+1) - 1) // (p - 1)
        p += 2
    if temp > 1:
        result *= (temp + 1)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Known solution
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Use sieve to compute sigma(k) for all k up to N efficiently
    # sigma[n] = sum of divisors of n
    # O(N log N) sieve vs O(N sqrt(N)) trial division
    
    N = 2000000
    sigma_arr = [0] * (N + 1)
    for d in range(1, N + 1):
        for m in range(d, N + 1, d):
            sigma_arr[m] += d
    
    groups = defaultdict(list)
    
    for k in range(1, N + 1):
        if time.time() > deadline:
            break
        M = k * sigma_arr[k]
        if M <= 10**18:
            groups[M].append(k)
            if len(groups[M]) > best_count:
                best_count = len(groups[M])
                best_M = M
                best_k = groups[M][:]
    
    return {'M': int(best_M), 'k': [int(x) for x in best_k]}
