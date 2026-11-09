# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using m^2 * b where b is squarefree divisor of m."""
    spf = list(range(isqrt(limit) + 2))
    for i in range(2, isqrt(isqrt(limit)) + 1):
        if spf[i] == i:
            for j in range(i*i, len(spf), i):
                if spf[j] == j:
                    spf[j] = i
    
    powerful_set = set([1])
    sqrt_limit = isqrt(limit)
    
    for m in range(1, sqrt_limit + 1):
        temp = m
        primes = []
        while temp > 1:
            if temp <= len(spf):
                p = spf[temp]
            else:
                p = temp
            primes.append(p)
            while temp % p == 0:
                temp //= p
        
        n_primes = len(primes)
        for mask in range(1 << n_primes):
            b = 1
            for i in range(n_primes):
                if mask & (1 << i):
                    b *= primes[i]
            if b <= limit // (m * m):
                powerful_set.add(m * m * b)
    
    return sorted(powerful_set)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # 5 * 10^8 gives ~7600 powerful numbers, ~29M pairs.
    # This should run in ~30-60s.
    M = 5 * 10**8
    powerful = _generate_powerful(M)
    K = len(powerful)
    
    sums_count = defaultdict(int)
    P = powerful
    sc = sums_count
    
    for i in range(K):
        pi = P[i]
        for j in range(i, K):
            sc[pi + P[j]] += 1
            
    if not sums_count:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    best_N = max(sums_count, key=sums_count.get)
    pw_set = set(powerful)
    best_pairs = []
    for a in P:
        if a > best_N // 2:
            break
        b = best_N - a
        if b >= a and b in pw_set:
            best_pairs.append([a, b])
            
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
