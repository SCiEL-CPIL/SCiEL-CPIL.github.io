import numpy as np

def construct():
    N = 123457
    
    # Compute Largest Prime Factor for each number up to N
    lpf = np.ones(N + 1, dtype=np.int32)
    
    # Sieve
    for i in range(2, N + 1):
        if lpf[i] == 1:
            lpf[i::i] = i

    # Vectorized scan
    a = np.arange(1, N, dtype=np.int32)
    b = N - a
    pa = lpf[a]
    pb = lpf[b]
    m = np.maximum(pa, pb)
    idx = np.argmin(m)
    
    best_a = int(a[idx])
    best_b = int(b[idx])
    
    return [best_a, best_b]
