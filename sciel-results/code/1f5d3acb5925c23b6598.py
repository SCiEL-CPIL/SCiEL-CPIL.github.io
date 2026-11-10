import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Set a reasonable N. 10^7 is safe for memory (80MB) and speed.
    N = 10_000_000
    
    # 1. Compute phi using NumPy
    phi = np.arange(1, N + 1, dtype=np.int64)
    
    # Sieve primes up to sqrt(N)
    limit = int(np.sqrt(N))
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # Update phi
    for p in primes:
        # p is the prime value
        # We need to update phi[j] for all j divisible by p
        # phi[j] = phi[j] * (1 - 1/p)
        # In integer arithmetic: phi[j] -= phi[j] // p
        # Indices in 0-based array: j-1.
        # j is p, 2p, 3p...
        # Indices: p-1, 2p-1, 3p-1...
        # Step is p.
        phi[p-1::p] -= phi[p-1::p] // p
        
    if time_left() < 10.0:
        # Fallback: return a small valid result
        return {'start': 1, 'length': 1}

    # 2. Find longest run of distinct values
    best_k = 0
    best_s = 1
    seen = set()
    left = 0
    
    # We can stop if we reach 200
    target_k = 200
    
    for right in range(N):
        val = phi[right]
        
        # Remove from left until val is unique in window
        while val in seen:
            seen.remove(phi[left])
            left += 1
            
        seen.add(val)
        
        current_k = right - left + 1
        if current_k > best_k:
            best_k = current_k
            best_s = left + 1 # Convert to 1-based index
            
            if best_k >= target_k:
                break
        
        if time_left() < 5.0:
            break

    # Ensure validity
    if best_k < 1:
        best_k = 1
        best_s = 1
        
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
        
    if best_k < 1:
        best_k = 1
        
    return {'start': best_s, 'length': best_k}
