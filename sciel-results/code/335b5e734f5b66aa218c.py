import time
from math import isqrt
import numpy as np

def construct():
    start_time = time.time()
    timeout = 115  # safety margin before 120s
    
    # Target M - increased to explore larger N
    M = 100_000_000
    
    powerful = set()
    max_y = int(M ** (1/3)) + 1
    
    for y in range(1, max_y + 1):
        y3 = y ** 3
        if y3 > M:
            break
        max_x = isqrt(M // y3)
        for x in range(1, max_x + 1):
            p = x * x * y3
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
    
    for i in range(n_list):
        if time.time() - start_time > timeout - 15:
            break
        # For row i, sums are a[i] + a[i], a[i] + a[i+1], ...
        sums = arr[i:] + arr[i]
        # Use np.add.at for atomic accumulation
        np.add.at(counts, sums, 1)
    
    if counts.max() == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    best_N = int(np.argmax(counts))
    
    # Reconstruct pairs
    pairs = []
    for a in powerful_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
