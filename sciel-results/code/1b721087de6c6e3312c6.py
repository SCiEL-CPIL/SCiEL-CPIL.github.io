import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    N = 50000000
    
    # phi array
    phi = np.arange(N + 1, dtype=np.int32)
    
    # last_seen array
    # Stores the last index where a phi value was seen.
    # 0 means not seen.
    last_seen = np.zeros(N + 1, dtype=np.int32)
    
    # Sieve
    limit = int(np.sqrt(N))
    for i in range(2, limit + 1):
        if phi[i] == i:
            phi[i::i] -= phi[i::i] // i
            
    if time.time() - start_time > deadline - 10:
        return {'start': 1, 'length': 1}

    best_k = 1
    best_s = 1
    start = 1
    
    # Scan
    for i in range(1, N + 1):
        val = phi[i]
        prev_idx = last_seen[val]
        if prev_idx >= start:
            start = prev_idx + 1
        last_seen[val] = i
        
        current_len = i - start + 1
        if current_len > best_k:
            best_k = current_len
            best_s = start
            if best_k >= 200:
                break
                
        if time.time() - start_time > deadline - 2:
            break

    return {'start': int(best_s), 'length': int(best_k)}
