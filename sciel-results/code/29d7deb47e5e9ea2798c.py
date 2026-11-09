import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin (90s kill limit)
    
    N = 1000000
    limit = N + 1
    
    lpf = np.zeros(limit + 1, dtype=np.int64)
    
    # NumPy sieve for Largest Prime Factor
    # Check time less frequently to reduce overhead
    sieve_complete = True
    for i in range(2, limit + 1):
        if i % 10000 == 0 and time.time() > deadline:
            sieve_complete = False
            break
        if lpf[i] == 0:
            lpf[i::i] = i
    
    if not sieve_complete:
        # Fallback: linear scan with partial sieve
        best_val = float('inf')
        best_n = 10000
        for n in range(10000, N + 1):
            a = int(lpf[n])
            b = int(lpf[n+1])
            if a == 0 or b == 0:
                continue
            val = a if a > b else b
            if val < best_val:
                best_val = val
                best_n = n
        return best_n
    
    # Vectorized scan for minimum max prime factor
    vals = np.maximum(lpf[10000:N+1], lpf[10001:N+2])
    min_idx = int(np.argmin(vals))
    best_n = 10000 + min_idx
    
    return best_n
