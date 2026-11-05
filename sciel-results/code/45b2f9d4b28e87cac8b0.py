import math
import time
import random

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
    LIMIT = 110.0  # Safety margin for 120s limit
    MAX_S = 10**12
    MAX_K = 200
    
    primes = PRIMES
    primes_len = len(primes)
    
    def phi(x):
        if x == 1:
            return 1
        result = x
        m = x
        # Check 2 separately
        if m % 2 == 0:
            result -= result // 2
            while m % 2 == 0:
                m //= 2
        # Check odd primes
        for i in range(1, primes_len):
            p = primes[i]
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
    
    # We will scan large contiguous blocks
    # Estimate speed: ~5-20us per phi
    # Let's try block size of 200,000
    BLOCK_SIZE = 200000
    
    while time.time() - start_time < LIMIT:
        # Pick a random block start
        # Ensure we don't exceed MAX_S
        # We need s + BLOCK_SIZE - 1 <= MAX_S
        max_start = MAX_S - BLOCK_SIZE + 1
        if max_start < 1:
            max_start = 1
            
        s_start = random.randint(1, max_start)
        
        left = 0
        seen = {}
        
        # Scan the block
        # We iterate right from 0 to BLOCK_SIZE - 1
        # Actual number is s_start + right
        for right in range(BLOCK_SIZE):
            current_num = s_start + right
            val = phi(current_num)
            
            if val in seen:
                prev_idx = seen[val]
                if prev_idx >= left:
                    left = prev_idx + 1
            
            seen[val] = right
            
            current_len = right - left + 1
            if current_len > best_k:
                best_k = current_len
                best_s = s_start + left
                if best_k >= MAX_K:
                    return {'start': best_s, 'length': MAX_K}
                    
            # Periodic time check inside the loop to avoid overshooting
            if right % 10000 == 0 and time.time() - start_time > LIMIT - 5:
                break

    return {'start': best_s, 'length': best_k}
