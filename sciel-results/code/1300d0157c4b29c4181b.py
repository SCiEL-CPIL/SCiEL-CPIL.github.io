import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Structural change: Use chunked pairwise sum computation with np.bincount
    # to handle larger L without O(M^2) memory.
    
    L = 5 * 10**7
    
    # Generate powerful numbers
    powerfuls = set()
    max_b = int(L ** (1.0/3.0)) + 2
    for b in range(1, max_b + 1):
        b3 = b * b * b
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerfuls.add(n)
    
    powerfuls_list = sorted(powerfuls)
    M = len(powerfuls_list)
    
    elapsed_gen = time.time() - start_time
    
    if time.time() > deadline - 40:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    P = np.array(powerfuls_list, dtype=np.int32)
    
    # Chunked computation to avoid O(M^2) memory
    max_sum = 2 * L
    counts = np.zeros(max_sum + 1, dtype=np.int32)
    
    chunk_size = 2000
    
    for i_start in range(0, M, chunk_size):
        i_end = min(i_start + chunk_size, M)
        chunk = P[i_start:i_end, None] + P[None, :]
        flat = chunk.ravel()
        chunk_counts = np.bincount(flat, minlength=max_sum + 1)
        counts += chunk_counts[:max_sum + 1]
        
        if time.time() > deadline - 30:
            break
    
    elapsed_count = time.time() - start_time
    
    if time.time() > deadline - 10:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Find N with maximum unordered pair count
    powerfuls_set = set(powerfuls_list)
    
    k = min(200, len(counts))
    top_indices = np.argpartition(counts, -k)[-k:]
    
    best_N = 2
    best_unordered = 0
    
    for N in top_indices:
        N = int(N)
        if N < 2:
            continue
        ordered = int(counts[N])
        if N % 2 == 0 and (N // 2) in powerfuls_set:
            unordered = (ordered + 1) // 2
        else:
            unordered = ordered // 2
        if unordered > best_unordered:
            best_unordered = unordered
            best_N = N
    
    # Extract pairs
    pairs = []
    for a in powerfuls_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerfuls_set:
            pairs.append([a, b])
    
    pairs.sort()
    
    if not pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N, 'pairs': pairs}
