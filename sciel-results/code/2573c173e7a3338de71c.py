import numpy as np
from math import isqrt

def construct():
    import time
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # 1. Generate powerful numbers up to L
    L = 10**8
    # Square-free m up to cbrt(L)
    max_m = int(L ** (1/3)) + 2
    square_free = []
    for m in range(1, max_m + 1):
        is_sf = True
        p = 2
        temp = m
        while p * p <= temp:
            if temp % (p * p) == 0:
                is_sf = False
                break
            p += 1
        if is_sf:
            square_free.append(m)
            
    powerful = set()
    for m in square_free:
        m3 = m ** 3
        max_k = isqrt(L // m3)
        for k in range(1, max_k + 1):
            val = k * k * m3
            if val <= L:
                powerful.add(val)
    
    powerful_list = sorted(powerful)
    
    # 2. FFT approach
    # We want to find N in [0, 2*L] with max representations
    size = 1
    while size < 2 * L + 1:
        size *= 2
        
    A = np.zeros(size, dtype=np.float64)
    for p in powerful_list:
        A[p] = 1.0
        
    # FFT
    F = np.fft.rfft(A)
    F2 = F * F
    I = np.fft.irfft(F2, n=size)
    
    # Find max
    # I[n] is the count of pairs (a,b) with a+b=n
    # We only care about n <= 2*L
    max_idx = np.argmax(I[:2*L+1])
    best_N = max_idx
    max_count_ordered = int(I[best_N])
    
    # 3. Extract pairs
    pairs = []
    for a in powerful_list:
        if a > best_N:
            break
        b = best_N - a
        if b in powerful:
            if a <= b:
                pairs.append([a, b])
    
    # Verify time
    if time.time() > deadline:
         # Fallback to a simple known good N if needed, but this should be fast
         pass
         
    return {'N': best_N, 'pairs': pairs}
