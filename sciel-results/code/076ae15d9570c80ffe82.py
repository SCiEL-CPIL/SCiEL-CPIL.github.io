import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin for the 90s limit
    
    max_val = 1000001
    # We need lpf for indices 0..max_val (we access lpf up to 1000001)
    # lpf[i] will store the largest prime factor of i
    # For i=0,1 lpf will be 0 (undefined but we don't care)
    lpf = np.zeros(max_val + 1, dtype=np.int32)
    
    # Sieve for Largest Prime Factor
    # For each prime p, assign p to all multiples of p
    # We need to identify primes. If lpf[p] == 0, p is prime (for p>=2)
    
    for p in range(2, max_val + 1):
        if lpf[p] == 0:
            # p is prime
            lpf[p::p] = p
            
            # Time check
            if p % 50000 == 0:
                if time.time() > deadline:
                    # If we timeout, we cannot safely return a result based on incomplete LPF.
                    # We must return a known good result or a safe default.
                    # 10000 is a safe default.
                    return 10000

    if time.time() > deadline:
        return 10000

    # We want to minimize max(lpf[n], lpf[n+1]) for n in [10000, 1000000]
    # n ranges from 10000 to 1000000
    # lpf[n] corresponds to indices 10000 to 1000000
    # lpf[n+1] corresponds to indices 100001 to 1000001
    
    # Vectorized calculation of scores
    # lpf[10000:1000001] has length 990001 (indices 10000..1000000)
    # lpf[100001:1000002] has length 990001 (indices 100001..1000001)
    
    # Note: The array size is max_val + 1 = 1000002.
    # Index 1000001 is valid.
    
    lpf_n = lpf[10000:1000001]
    lpf_np1 = lpf[100001:1000002]
    
    scores = np.maximum(lpf_n, lpf_np1)
    
    if len(scores) == 0:
        return 10000
        
    min_score_idx = np.argmin(scores)
    best_n = 10000 + min_score_idx
    
    return int(best_n)
