import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N_MIN = 10000
    N_MAX = 1000000
    N_MAX_PLUS_1 = N_MAX + 1
    SIZE = N_MAX_PLUS_1 + 1  # 1000002
    
    # Check deadline
    if time.time() > deadline:
        return N_MIN # Fallback, though unlikely to happen this early
        
    # Compute Largest Prime Factor (LPF) using a sieve
    # lpf[i] will store the largest prime factor of i
    lpf = np.zeros(SIZE, dtype=np.int32)
    
    # Sieve of Eratosthenes variant for LPF
    # Iterate through potential primes
    for p in range(2, SIZE):
        if lpf[p] == 0:
            # p is prime
            # Mark all multiples of p
            # We want the LARGEST prime factor, so we overwrite.
            # Since we iterate p in increasing order, the last write is by the largest prime.
            lpf[p:SIZE:p] = p
            
    if time.time() > deadline:
        # If we ran out of time during sieve (unlikely for 1e6), 
        # we can't compute exact LPF. 
        # But we need a valid n. 
        # We could return a heuristic guess, but the sieve is fast.
        return N_MIN

    # Calculate max(LPF[n], LPF[n+1]) for all n in range
    # n ranges from N_MIN to N_MAX
    # LPF[n] is lpf[N_MIN : N_MAX+1]
    # LPF[n+1] is lpf[N_MIN+1 : N_MAX+2]
    
    lpf_n = lpf[N_MIN:N_MAX+1]
    lpf_n1 = lpf[N_MIN+1:N_MAX+2]
    
    max_lpf = np.maximum(lpf_n, lpf_n1)
    
    if time.time() > deadline:
        return N_MIN

    # Find the index of the minimum value
    min_idx = np.argmin(max_lpf)
    best_n = N_MIN + min_idx
    
    return int(best_n)
