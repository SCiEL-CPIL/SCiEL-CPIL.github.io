import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Precompute powerful numbers up to L
    L = 5 * 10**8
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
    powerfuls_set = powerfuls
    
    if time.time() > deadline - 60:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Convert to numpy
    P = np.array(powerfuls_list, dtype=np.int64)
    max_sum = 2 * L
    
    # Use chunked bincount with int32 to save memory
    counts = np.zeros(max_sum + 1, dtype=np.int32)
    chunk_size = 1000  # Adjust based on memory
    
    for i_start in range(0, M, chunk_size):
        i_end = min(i_start + chunk_size, M)
        chunk = P[i_start:i_end, None] + P[None, :]
        flat = chunk.ravel()
        # Filter sums <= max_sum (all are, since a,b <= L)
        chunk_counts = np.bincount(flat, minlength=max_sum + 1)
        counts += chunk_counts[:max_sum + 1]
        
        if time.time() > deadline - 40:
            break
    
    # Find top candidates
    k = min(100, len(counts))
    top_indices = np.argpartition(counts, -k)[-k:]
    
    best_N = 2
    best_pairs = [[1, 1]]
    
    # Sort top indices by count descending
    top_indices = top_indices[np.argsort(counts[top_indices])[::-1]]
    
    for N in top_indices[:50]:
        N = int(N)
        if N < 2:
            continue
        pairs = []
        for a in powerfuls_list:
            if a > N // 2:
                break
            b = N - a
            if b in powerfuls_set:
                pairs.append([a, b])
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
            best_N = N
            
        if time.time() > deadline - 30:
            break
    
    if not best_pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_pairs.sort()
    
    return {'N': best_N, 'pairs': best_pairs}
