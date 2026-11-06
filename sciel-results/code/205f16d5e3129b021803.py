"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer for 120s limit
    
    # Generate powerful numbers up to L
    L = 120_000_000  # ~1.2 * 10^8
    
    powerfuls = set()
    max_b = int(L ** (1/3)) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerfuls.add(n)
    
    powerfuls_list = sorted(powerfuls)
    n_pows = len(powerfuls_list)
    
    # Create indicator array for FFT-based sum counting
    max_sum = 2 * L
    # Use next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < max_sum + 1:
        fft_size *= 2
    
    # Create indicator array
    ind = np.zeros(fft_size, dtype=np.float64)
    for p in powerfuls_list:
        ind[p] = 1.0
    
    # FFT convolution
    fft_ind = np.fft.rfft(ind)
    conv = np.fft.irfft(fft_ind * fft_ind, n=fft_size)
    conv = np.round(conv).astype(np.int64)
    
    # conv[s] = number of ordered pairs (a,b) with a+b=s where both are powerful
    # For unordered pairs a<=b:
    # unordered_count[s] = (conv[s] + eq[s]) // 2 where eq[s]=1 if s even and s/2 in powerfuls
    pows_set = set(powerfuls_list)
    
    # Create eq array: eq[s] = 1 if s is even and s//2 is powerful
    eq = np.zeros(max_sum + 1, dtype=np.int64)
    for p in powerfuls_list:
        s = 2 * p
        if s <= max_sum:
            eq[s] = 1
    
    # Unordered count
    unordered = (conv[:max_sum+1] + eq) // 2
    
    # Find best N using numpy
    best_N = int(np.argmax(unordered[2:])) + 2
    best_count = int(unordered[best_N])
    
    if best_count == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Collect pairs for best_N
    pairs = []
    for a in powerfuls_list:
        if 2 * a > best_N:
            break
        b_val = best_N - a
        if b_val in pows_set and b_val >= a:
            pairs.append([a, b_val])
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
