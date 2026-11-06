import time
import numpy as np
from math import isqrt

def _is_powerful(n):
    if n == 1:
        return True
    if n < 1:
        return False
    x = n
    d = 2
    while d * d <= x:
        if x % d == 0:
            cnt = 0
            while x % d == 0:
                x //= d
                cnt += 1
            if cnt < 2:
                return False
        d += 1
    if x > 1:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    L = 8 * 10**7
    powerful_set = set()
    max_b = int(L ** (1.0/3.0)) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerful_set.add(n)
    
    powerful_list = sorted([n for n in powerful_set if _is_powerful(n)])
    elapsed = time.time() - start_time
    print(f"Generated {len(powerful_list)} powerful numbers up to {L}, took {elapsed:.1f}s")
    
    if time.time() > deadline - 50:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    powerful_set_list = set(powerful_list)
    max_N = 2 * L
    
    arr = np.zeros(max_N + 1, dtype=np.float64)
    for p in powerful_list:
        if p <= max_N:
            arr[p] = 1.0
    
    from scipy.signal import fftconvolve
    counts = fftconvolve(arr, arr, mode='full')
    
    elapsed = time.time() - start_time
    print(f"FFT done, took {elapsed:.1f}s total")
    
    if time.time() > deadline - 20:
        # Can't safely extract pairs, return what we have
        best_N = 2
        best_count = 0
        # Quick scan of top values
        top_indices = np.argsort(counts)[::-1][:100]
        for N in top_indices:
            N = int(N)
            if N < 2 or N > max_N:
                continue
            ordered = int(round(counts[N]))
            if N % 2 == 0 and (N // 2) in powerful_set_list:
                unordered = (ordered + 1) // 2
            else:
                unordered = ordered // 2
            if unordered > best_count:
                best_count = unordered
                best_N = N
        pairs = []
        for a in powerful_list:
            if a > best_N // 2:
                break
            b = best_N - a
            if b in powerful_set_list:
                pairs.append([a, b])
        pairs.sort()
        return {'N': best_N, 'pairs': pairs}
    
    best_N = 2
    best_count = 0
    for N in range(2, min(max_N + 1, len(counts))):
        ordered = int(round(counts[N]))
        if N % 2 == 0 and (N // 2) in powerful_set_list:
            unordered = (ordered + 1) // 2
        else:
            unordered = ordered // 2
        if unordered > best_count:
            best_count = unordered
            best_N = N
    
    print(f"Best N = {best_N}, count = {best_count}, elapsed {time.time()-start_time:.1f}s")
    
    pairs = []
    for a in powerful_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set_list:
            pairs.append([a, b])
    
    pairs.sort()
    
    return {'N': best_N, 'pairs': pairs}
