"""Sieve-based search for M with many k solutions."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit
    
    # We use a sieve to compute sigma(k) for k up to N.
    # This ensures all k are distinct and processed exactly once.
    N = 2_000_000  # 2 million is fast and safe
    
    # 1. Sieve for smallest prime factor (SPF)
    spf = list(range(N + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute sigma(n) for all n
    sigma = [0] * (N + 1)
    sigma[1] = 1
    for i in range(2, N + 1):
        n = i
        res = 1
        while n > 1:
            p = spf[n]
            power_sum = 1
            power = 1
            while n % p == 0:
                n //= p
                power *= p
                power_sum += power
            res *= power_sum
        sigma[i] = res
        
    # 3. Map M -> list of k
    m_to_ks = defaultdict(list)
    max_m_allowed = 10**18
    
    for k in range(1, N + 1):
        M = k * sigma[k]
        if M <= max_m_allowed:
            m_to_ks[M].append(k)
            
    # 4. Find best M
    best_m = 1
    best_ks = [1]
    best_count = 1
    
    for M, ks in m_to_ks.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_m = M
            best_ks = ks
            if best_count >= 1000: # Max allowed
                break
                
    # Ensure types are correct
    best_m = int(best_m)
    best_ks = [int(x) for x in best_ks[:1000]]
    
    return {'M': best_m, 'k': best_ks}
# EVOLVE-BLOCK-END
