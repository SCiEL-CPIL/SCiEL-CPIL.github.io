import time
import numpy as np
from math import isqrt

def _is_powerful(n):
    if n <= 1:
        return True
    d = 2
    while d * d <= n:
        if n % d == 0:
            cnt = 0
            while n % d == 0:
                n //= d
                cnt += 1
            if cnt == 1:
                return False
        d += 1
    if n > 1:
        return False
    return True

def _generate_powerful(limit):
    powerful = set()
    max_y = int(limit ** (1/3)) + 2
    for y in range(1, max_y + 1):
        y3 = y ** 3
        if y3 > limit:
            break
        max_x = isqrt(limit // y3)
        for x in range(1, max_x + 1):
            n = x * x * y3
            if n <= limit:
                powerful.add(n)
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = 135
    
    M = 10_000_000_000
    powerful_list = _generate_powerful(M)
    K = len(powerful_list)
    powerful_arr = np.array(powerful_list, dtype=np.int64)
    powerful_set = set(powerful_list)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 60:
        M = 1_000_000_000
        powerful_list = _generate_powerful(M)
        K = len(powerful_list)
        powerful_arr = np.array(powerful_list, dtype=np.int64)
        powerful_set = set(powerful_list)
    
    counts = {}
    for i in range(K):
        if time.time() - start_time > deadline - 20:
            break
        a = powerful_arr[i]
        sums = powerful_arr[i:] + a
        unique_sums, cnts = np.unique(sums, return_counts=True)
        for idx in range(len(unique_sums)):
            s = int(unique_sums[idx])
            counts[s] = counts.get(s, 0) + int(cnts[idx])
    
    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_N = max(counts, key=counts.get)
    
    # Extend powerful set for verification
    if best_N > M:
        M_ext = min(2 * M, 10**12)
        powerful_ext = _generate_powerful(M_ext)
        powerful_set_ext = set(powerful_ext)
    else:
        powerful_set_ext = powerful_set
    
    pairs = []
    for a in powerful_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set_ext:
            pairs.append([a, b])
    
    verified_pairs = []
    for a, b in pairs:
        if _is_powerful(a) and _is_powerful(b):
            verified_pairs.append([a, b])
    
    if not verified_pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N, 'pairs': verified_pairs}
