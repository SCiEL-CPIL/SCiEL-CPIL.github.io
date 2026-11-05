import math
import time

def construct():
    start_time = time.time()
    LIMIT = 145.0 # seconds, leaving 5s margin for 150s limit
    MAX_S = 10**12
    MAX_K = 200

    # 1. Generate primes up to 10^6
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # 2. Efficient phi calculation
    def phi(x):
        if x == 1:
            return 1
        result, m = x, x
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                result -= result // p
                while m % p == 0:
                    m //= p
        if m > 1:
            result -= result // m
        return result

    # 3. Direct Search for k=200
    # Start from a safe high number, step down.
    # If we find a valid run of 200, return it.
    
    def longest_distinct_run(vals):
        n = len(vals)
        if n == 0:
            return 0, 0
        max_len = 1
        max_start = 0
        cur_start = 0
        seen = {}
        for i in range(n):
            v = vals[i]
            if v in seen and seen[v] >= cur_start:
                cur_start = seen[v] + 1
            seen[v] = i
            cur_len = i - cur_start + 1
            if cur_len > max_len:
                max_len = cur_len
                max_start = cur_start
        return max_len, max_start

    start_s = MAX_S - 10000
    step = 1000
    best_k = 1
    best_s = 1
    
    while True:
        if time.time() - start_time > LIMIT:
            break
            
        if start_s < MAX_K:
            break
            
        # Check range [start_s, start_s + MAX_K - 1]
        phis = []
        for i in range(MAX_K):
            n = start_s + i
            phis.append(phi(n))
            
        # Check if all 200 are distinct
        if len(set(phis)) == MAX_K:
            return {'start': start_s, 'length': MAX_K}
        else:
            # Find the longest run of distinct phis in this window
            k, offset = longest_distinct_run(phis)
            if k > best_k:
                best_k = k
                best_s = start_s + offset
            start_s -= step

    # If we exit loop without finding 200, return the best found
    return {'start': best_s, 'length': best_k}
