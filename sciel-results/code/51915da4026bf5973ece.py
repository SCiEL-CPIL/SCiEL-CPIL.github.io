import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 1000001
    lpf = np.arange(N + 1, dtype=np.int32)
    
    # Sieve for largest prime factor
    # We only need to check primes up to N, but most updates happen for small p.
    # If we iterate p from 2 to N, and lpf[p] == p, it's a prime.
    # We can stop early? No, we need to mark all multiples.
    # But for p > sqrt(N), the only multiple is p itself (and p*p > N).
    # So for p > sqrt(N), lpf[p] is already p (if prime) or composite.
    # If composite, lpf[p] < p.
    # So we can just iterate to N.
    
    # Optimization: only iterate p where lpf[p] == p (primes)
    # But checking lpf[p]==p for all p is O(N).
    # The inner loop is O(N/p).
    # Total O(N log log N).
    
    for p in range(2, N + 1):
        if lpf[p] == p:  # p is prime
            # Mark all multiples of p with p
            # This overwrites previous smaller prime factors
            lpf[p::p] = p
            
        if time.time() > deadline:
            break

    # Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    # lpf is 0-indexed. n is in [10000, 1000000].
    # We need max(lpf[n], lpf[n+1]) for n in 10000..1000000.
    # This corresponds to indices 10000..1000000 in lpf.
    # lpf[n+1] corresponds to indices 10001..1000001.
    
    # Create arrays for the range
    # n values: 10000 to 1000000
    # lpf_n = lpf[10000:1000001]
    # lpf_n1 = lpf[10001:1000002]
    
    lpf_n = lpf[10000:1000001]
    lpf_n1 = lpf[10001:1000002]
    
    # Calculate max
    max_lpf = np.maximum(lpf_n, lpf_n1)
    
    # Find index of minimum
    min_idx = np.argmin(max_lpf)
    best_n = 10000 + min_idx
    
    return best_n
