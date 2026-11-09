import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate primes up to 10^6
    limit = 10**6
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    primes = np.array(np.nonzero(sieve)[0])
    
    best_k = 1
    best_s = 1
    
    block_size = 10000
    # Start scanning from a high value to leverage sparsity
    # The prompt suggests high-range totients are less prone to collisions
    current_s = 10**11
    
    while time.time() < deadline:
        if current_s + block_size > 10**12:
            break
            
        # Compute phi for block
        vals = np.arange(current_s, current_s + block_size, dtype=np.int64)
        
        # Sieve phi
        max_p = int(np.sqrt(current_s + block_size))
        # Filter primes
        relevant_primes = primes[primes <= max_p]
        
        for p in relevant_primes:
            start_idx = (-current_s) % p
            if start_idx >= block_size:
                continue
            indices = np.arange(start_idx, block_size, p)
            vals[indices] -= vals[indices] // p
            
        # Handle remaining prime factors
        mask = vals > 1
        vals[mask] -= 1
        
        # Find longest run of distinct values
        seen = set()
        left = 0
        for right in range(block_size):
            v = vals[right]
            if v in seen:
                while vals[left] != v:
                    seen.remove(vals[left])
                    left += 1
                left += 1
            seen.add(v)
            current_len = right - left + 1
            if current_len > best_k:
                best_k = current_len
                best_s = current_s + left
        
        current_s += block_size
        
    return {'start': best_s, 'length': best_k}
