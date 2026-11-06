import numpy as np

def construct():
    N = 10_000_000
    
    phi = np.arange(N + 1, dtype=np.int64)
    
    # NumPy Sieve
    for i in range(2, N + 1):
        if phi[i] == i:
            # Update multiples of i
            # phi[j] = phi[j] * (1 - 1/i)
            # To avoid floating point, use integer arithmetic
            # phi[j] -= phi[j] // i
            phi[i::i] -= phi[i::i] // i

    last_seen = {}
    start = 1
    max_len = 0
    max_start = 1
    
    for i in range(1, N + 1):
        val = phi[i]
        if val in last_seen and last_seen[val] >= start:
            start = last_seen[val] + 1
        last_seen[val] = i
        
        current_len = i - start + 1
        if current_len > max_len:
            max_len = current_len
            max_start = start
            
    return {'start': max_start, 'length': max_len}
