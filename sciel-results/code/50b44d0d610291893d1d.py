import time
import math
import numpy as np

def construct():
    start = time.time()
    deadline = start + 140  # safety margin
    
    def generate_powerful(L):
        powerful = set()
        b = 1
        b3 = 1
        while b3 <= L:
            a_max = math.isqrt(L // b3)
            for a in range(1, a_max + 1):
                powerful.add(a * a * b3)
            b += 1
            b3 = b ** 3
        return sorted(powerful)
    
    # Choose L
    L = 3 * 10**10
    
    P = generate_powerful(L)
    n = len(P)
    P_arr = np.array(P, dtype=np.int64)
    
    # Count pairs
    # Generate all sums P[i] + P[j] for i <= j
    sums_chunks = []
    for i in range(n):
        if time.time() > deadline - 30:
            # Fallback: use what we have
            break
        sums_chunks.append(P_arr[i] + P_arr[i:])
    
    if not sums_chunks:
        # No pairs found, return a default
        return {'N': 2, 'pairs': [[1, 1]]}
    
    sums = np.concatenate(sums_chunks)
    
    # Sort and count
    sums_sorted = np.sort(sums)
    unique, counts = np.unique(sums_sorted, return_counts=True)
    
    max_idx = np.argmax(counts)
    best_N = int(unique[max_idx])
    best_count = int(counts[max_idx])
    
    # Enumerate all pairs for best_N
    P_set = set(P)
    pairs = []
    for a in P:
        if a * 2 > best_N:
            break
        b = best_N - a
        if b in P_set:
            pairs.append([a, b])
    
    return {'N': best_N, 'pairs': pairs}
