import math
import time
import bisect
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    L = 10**12
    
    # Generate powerful numbers up to L
    # Powerful numbers are x^2 * y^3 where y is square-free.
    # Max y is L^(1/3) = 10000.
    # Max x is sqrt(L / y^3).
    
    # Primes up to 10000 for square-free check
    limit_primes = int(L ** (1.0/3.0)) + 2
    is_prime = [True] * (limit_primes + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_primes ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_primes + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    def is_square_free(n):
        if n < 2:
            return True
        for p in primes:
            if p * p > n:
                break
            if n % (p * p) == 0:
                return False
        return True
        
    powerful_set = set()
    max_b = int(L ** (1.0/3.0))
    for b in range(1, max_b + 1):
        if not is_square_free(b):
            continue
        b3 = b ** 3
        if b3 > L:
            break
        max_a = int(math.isqrt(L // b3))
        for a in range(1, max_a + 1):
            val = a * a * b3
            powerful_set.add(val)
            
    powerful_list = sorted(powerful_set)
    M = len(powerful_list)
    
    # Count representations
    counts = defaultdict(int)
    best_N = 0
    best_count = 0
    
    # Optimize local variables
    p_list = powerful_list
    counts_get = counts.get
    
    for i in range(M):
        ai = p_list[i]
        if time.time() > deadline:
            break
        # We can break if ai > L? No, ai is always <= L.
        # We iterate j from i to M-1
        for j in range(i, M):
            b = p_list[j]
            N = ai + b
            c = counts_get(N, 0) + 1
            counts[N] = c
            if c > best_count:
                best_count = c
                best_N = N
                
    if best_N == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Retrieve pairs
    p_set = set(powerful_list)
    pairs = []
    for a in powerful_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in p_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
