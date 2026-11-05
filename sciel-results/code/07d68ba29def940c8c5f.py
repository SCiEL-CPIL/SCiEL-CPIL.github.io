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
    
    # Segmented Sieve for Phi
    def phi_sieve(L, W):
        # Computes phi for L, L+1, ..., L+W-1
        # Returns list of length W
        vals = [L + i for i in range(W)]
        phis = [L + i for i in range(W)]
        
        # Handle 1 separately if in range
        if L == 1:
            phis[0] = 1
            # We can skip sieving 1, but the loop logic below might touch it?
            # Primes are >= 2. 1 is not a multiple of any prime.
            # So phis[0] stays 1 if we initialize correctly?
            # Wait, if L=1, vals[0]=1. 
            # In the loop, we check multiples of p. 1 is not a multiple of p>=2.
            # So phis[0] remains 1. Correct.
            
        # Sieve with primes
        for p in primes:
            if p * p > L + W - 1:
                break
                
            # Find first multiple of p >= L
            # k = ceil(L / p)
            if L % p == 0:
                k = L
            else:
                k = L + p - (L % p)
            
            # Iterate through multiples
            for j in range(k, L + W, p):
                idx = j - L
                phis[idx] -= phis[idx] // p
        
        # Handle remaining prime factors
        for i in range(W):
            if vals[i] > 1:
                phis[i] -= phis[i] // vals[i]
                
        return phis

    best_k = 1
    best_s = 1
    
    # We will scan large contiguous blocks
    # Estimate speed: Sieve is very fast. 
    # Let's try block size of 50,000. 
    # 50,000 * log(log(10^12)) ~ 50,000 * 2.5 ~ 125,000 ops.
    # Plus overhead. Should be < 1ms per block? 
    # Let's be conservative: 20,000.
    
    BLOCK_SIZE = 20000
    
    while time.time() - start_time < LIMIT:
        # Pick a random block start
        # Ensure we don't exceed MAX_S
        # We need s + BLOCK_SIZE - 1 <= MAX_S
        max_start = MAX_S - BLOCK_SIZE + 1
        if max_start < 1:
            max_start = 1
            
        s_start = random.randint(1, max_start)
        
        phis = phi_sieve(s_start, BLOCK_SIZE)
        
        left = 0
        seen = {}
        
        for right in range(BLOCK_SIZE):
            val = phis[right]
            
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
                    
        # Periodic time check
        if time.time() - start_time > LIMIT - 5:
            break

    return {'start': best_s, 'length': best_k}
