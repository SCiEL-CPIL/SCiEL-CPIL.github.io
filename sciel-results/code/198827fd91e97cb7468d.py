import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # L = 3 * 10^7
    # 2L = 6 * 10^7
    # Next power of 2 for FFT: 2^26 = 67,108,864
    # Memory is safe.
    L = 3 * 10**7
    
    # 1. Generate powerful numbers <= L
    powerful_set = set()
    max_y = int(L ** (1/3)) + 1
    
    sq_free = []
    for y in range(1, max_y + 1):
        is_sq_free = True
        temp_y = y
        d = 2
        while d * d <= temp_y:
            if temp_y % (d * d) == 0:
                is_sq_free = False
                break
            d += 1
        if is_sq_free:
            sq_free.append(y)
            
    for y in sq_free:
        y_cubed = y ** 3
        if y_cubed > L:
            continue
        max_x = int(math.isqrt(L // y_cubed))
        for x in range(1, max_x + 1):
            val = x * x * y_cubed
            powerful_set.add(val)
            
    if time.time() > deadline - 5:
        return {'N': 2, 'pairs': [[1, 1]]}

    P = sorted(powerful_set)
    
    # 2. FFT Convolution
    n_fft = 1
    while n_fft < 2 * L + 1:
        n_fft *= 2
        
    A = np.zeros(n_fft, dtype=np.float64)
    for p in P:
        A[p] = 1.0
        
    F = np.fft.fft(A)
    C = np.fft.ifft(F * F).real
    C = np.round(C).astype(np.int64)
    
    # 3. Find max count
    # Valid N range: 2 to 2L
    # C is 0-indexed.
    
    # Find max in valid range
    max_idx = np.argmax(C)
    best_N = int(max_idx)
    
    # Ensure best_N is valid
    if best_N < 2 or best_N > 2 * L:
        best_N = 2
        
    # 4. Retrieve pairs
    pairs = []
    powerful_set_lookup = powerful_set
    half_best = best_N // 2
    
    for a in P:
        if a > half_best:
            break
        b = best_N - a
        if b in powerful_set_lookup:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
