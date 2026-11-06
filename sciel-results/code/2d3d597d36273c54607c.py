import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    L = 10**8  # Upper bound for individual powerful numbers
    
    # 1. Generate powerful numbers up to L
    # Precompute primes up to sqrt(L) for factorization
    sqrt_L = int(math.isqrt(L))
    is_prime = [True] * (sqrt_L + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(sqrt_L)) + 1):
        if is_prime[i]:
            for j in range(i*i, sqrt_L + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    def is_powerful(n):
        if n == 1:
            return True
        x = n
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                cnt = 0
                while x % p == 0:
                    x //= p
                    cnt += 1
                if cnt < 2:
                    return False
        if x > 1:
            # x is a prime factor with exponent 1
            return False
        return True

    powerful_set = set()
    max_b = int(L ** (1.0/3.0)) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = int(math.isqrt(L // b3))
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                if is_powerful(n):
                    powerful_set.add(n)
    
    powerful_list = sorted(powerful_set)
    M = len(powerful_list)
    
    elapsed = time.time() - start_time
    if elapsed > 30:
        # Fallback if generation is too slow
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # 2. Count pairs
    # Using a dictionary to count sums
    # M is approx 10000 for L=10^8
    # M^2/2 = 50,000,000 operations
    # This might take 5-10 seconds in Python
    
    counts = defaultdict(int)
    # Optimization: local variables
    P = powerful_list
    for i in range(M):
        ai = P[i]
        for j in range(i, M):
            s = ai + P[j]
            counts[s] += 1
            
    # 3. Find best N
    best_N = 2
    best_count = 0
    for N, cnt in counts.items():
        if cnt > best_count:
            best_count = cnt
            best_N = N
            
    # 4. Reconstruct pairs
    pairs = []
    p_set = powerful_set
    for a in P:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in p_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
