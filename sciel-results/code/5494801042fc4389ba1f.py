import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def time_left():
        return deadline - time.time()
    
    def generate_powerful(limit):
        powerful = set()
        max_v = int(limit ** (1/3)) + 1
        is_sf = [True] * (max_v + 1)
        is_sf[0] = False
        is_sf[1] = True
        for p in range(2, int(math.isqrt(max_v)) + 1):
            if is_sf[p]:
                p2 = p * p
                for k in range(p2, max_v + 1, p2):
                    is_sf[k] = False
        for v in range(1, max_v + 1):
            if not is_sf[v]:
                continue
            v3 = v ** 3
            if v3 > limit:
                break
            max_u = int(math.isqrt(limit // v3))
            for u in range(1, max_u + 1):
                x = u * u * v3
                if x <= limit:
                    powerful.add(x)
        return sorted(powerful)
    
    M = 200000000
    P = generate_powerful(M)
    
    if time.time() > deadline:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    L = len(P)
    P_arr = np.array(P, dtype=np.int64)
    max_sum = 2 * M
    
    # Use numpy to compute all pair sums
    # We'll process in chunks to avoid memory issues
    hist = np.zeros(max_sum + 1, dtype=np.int32)
    
    # For each i, j >= i
    # Process i in chunks
    chunk_size = 10000
    
    for i_start in range(0, L, chunk_size):
        i_end = min(i_start + chunk_size, L)
        
        # For each i in this chunk, compute P_arr[i:] + P_arr[i]
        # and add to hist
        for i in range(i_start, i_end):
            sums = P_arr[i:] + P_arr[i]
            hist += np.bincount(sums, minlength=max_sum + 1)
        
        if time.time() > deadline - 10:
            break
    
    best_count = 0
    best_N = 0
    for s in range(2, max_sum + 1):
        c = hist[s]
        if c > best_count:
            best_count = c
            best_N = s
    
    if best_N == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    P_set = set(P)
    pairs = []
    for a in P:
        b = best_N - a
        if b < a:
            continue
        if b in P_set:
            pairs.append([a, b])
    
    return {'N': best_N, 'pairs': pairs}
