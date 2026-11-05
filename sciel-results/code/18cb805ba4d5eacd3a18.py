import math
import time
import random

# Precompute primes up to 10^6
def _get_primes(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, val in enumerate(is_prime) if val]

PRIMES = _get_primes(10**6)

def construct():
    start_time = time.time()
    LIMIT = 110.0 # Safety margin for 120s limit
    MAX_S = 10**12
    MAX_K = 200
    BLOCK_SIZE = 50000
    
    # Localize for speed
    primes = PRIMES
    
    def phi(x):
        if x == 1:
            return 1
        result = x
        m = x
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

    best_k = 1
    best_s = 1
    
    max_block_start = MAX_S - BLOCK_SIZE + 1
    
    while time.time() - start_time < LIMIT:
        # Pick a random block start
        # Ensure we don't exceed MAX_S
        s_start = random.randint(1, max_block_start)
        
        left = 0
        seen = {}
        
        # Scan the block
        for right in range(BLOCK_SIZE):
            current_num = s_start + right
            val = phi(current_num)
            
            if val in seen and seen[val] >= left:
                left = seen[val] + 1
            seen[val] = right
            
            current_len = right - left + 1
            if current_len > best_k:
                best_k = current_len
                best_s = s_start + left
                if best_k >= MAX_K:
                    return {'start': best_s, 'length': MAX_K}

    return {'start': best_s, 'length': best_k}
