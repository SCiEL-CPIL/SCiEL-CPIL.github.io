import time
from math import isqrt
import numpy as np

def construct():
    start_time = time.time()
    timeout = 145  # safety margin before 150s limit
    
    # Target M - increased to explore larger N for more representations
    M = 200_000_000
    
    powerful = set()
    max_y = int(M ** (1/3)) + 1
    
    for y in range(1, max_y + 1):
        y3 = y ** 3
        if y3 > M:
            break
        max_x = isqrt(M // y3)
        for x in range(1, max_x + 1):
            p = x * x * y3
            if p <= M:
                powerful.add(p)
    
    powerful_list = sorted(powerful)
    powerful_set = set(powerful_list)
    n_list = len(powerful_list)
    
    if n_list < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Use numpy for faster pairwise sum counting
    arr = np.array(powerful_list, dtype=np.int64)
    max_sum = 2 * M
    counts = np.zeros(max_sum + 1, dtype=np.int32)
    
    # Process in chunks to allow time checking between chunks
    chunk_size = 5000
    for i in range(0, n_list, chunk_size):
        if time.time() - start_time > timeout - 20:
            break
        end = min(i + chunk_size, n_list)
        for ii in range(i, end):
            sums = arr[ii:] + arr[ii]
            # Filter to valid range
            valid = (sums >= 0) & (sums <= max_sum)
            if valid.any():
                np.add.at(counts, sums[valid], 1)
    
    if counts.max() == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    best_N = int(np.argmax(counts))
    best_count = int(counts[best_N])
    
    # Reconstruct pairs
    pairs = []
    for a in powerful_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set:
            pairs.append([a, b])
    
    return {'N': best_N, 'pairs': pairs}
