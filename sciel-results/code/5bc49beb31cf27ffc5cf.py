import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    # We want to maximize the number of representations N = a + b
    # where a, b are powerful.
    # Heuristic suggests larger N is better.
    # We are limited by memory for FFT.
    # Let's try L = 10^8.
    # Number of powerful numbers ~ 2.6 * sqrt(10^8) ~ 26000.
    # FFT size needs to be > 2L.
    # Next power of 2 > 2*10^8 is 2^28 = 268,435,456.
    # Memory for complex128 array of size 2^28 is 2^28 * 16 bytes = 4.3 GB.
    # This might be too high.
    # Let's try L = 5 * 10^7.
    # 2L = 10^8. Next power of 2 is 2^27 = 134,217,728.
    # Memory: 134M * 16 bytes = 2.1 GB. Still high.
    # Let's try L = 2 * 10^7.
    # 2L = 4 * 10^7. Next power of 2 is 2^26 = 67,108,864.
    # Memory: 67M * 16 bytes = 1.07 GB. This is safer.
    
    L = 2 * 10**7
    
    # Generate powerful numbers up to L
    powerful_set = set()
    powerful_set.add(1)
    
    max_b = int(round(L ** (1/3))) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerful_set.add(n)
    
    powerful = sorted(powerful_set)
    n_powerful = len(powerful)
    
    if time.time() > deadline - 60.0:
        # Fallback to smaller L if generation was slow (unlikely)
        L = 5 * 10**6
        powerful_set = {1}
        max_b = int(round(L ** (1/3))) + 1
        for b in range(1, max_b + 1):
            b3 = b ** 3
            if b3 > L:
                break
            max_a = isqrt(L // b3)
            for a in range(1, max_a + 1):
                n = a * a * b3
                if n <= L:
                    powerful_set.add(n)
        powerful = sorted(powerful_set)
        n_powerful = len(powerful)

    # Create frequency array
    # Size must be at least 2*L to avoid circular convolution artifacts
    # We need to find N up to 2*L.
    fft_size = 1
    while fft_size < 2 * L + 1:
        fft_size *= 2
        
    # Check memory/time feasibility
    # If fft_size is too large, reduce L and restart? 
    # For now, assume L=2*10^7 is okay.
    
    f = np.zeros(fft_size, dtype=np.float64)
    for p in powerful:
        f[p] = 1.0
        
    # FFT
    f_fft = np.fft.fft(f)
    conv_fft = f_fft * f_fft
    conv = np.fft.ifft(conv_fft).real
    
    # conv[N] is the number of ordered pairs (a,b) with a+b=N
    # We need unordered pairs a <= b.
    
    # Identify the best N
    best_N = 0
    best_count = 0
    
    # We only care about N up to 2*L
    max_N = 2 * L
    
    # Optimization: Iterate and compute unordered count
    # unordered = (ordered + correction) // 2
    # correction is 1 if N is even and N//2 is in powerful_set, else 0
    
    # To speed up, we can just find the max ordered count first, 
    # but the correction term is small. 
    # Let's just compute for all N.
    
    # Create a boolean array for powerful numbers for O(1) lookup
    # powerful_bool[i] = 1 if i in powerful_set else 0
    # But we already have f which is 1.0 if powerful.
    # So f[N//2] > 0.5 checks if N//2 is powerful.
    
    # Vectorized approach for counting?
    # ordered = conv[2:max_N+1]
    # Ns = np.arange(2, max_N+1)
    # evens = Ns % 2 == 0
    # mids = Ns // 2
    # is_mid_powerful = f[mids] > 0.5  # Note: f is defined on 0..fft_size-1
    # correction = np.where(evens, is_mid_powerful, 0)
    # unordered = (ordered + correction) // 2
    # max_idx = np.argmax(unordered)
    # best_N = Ns[max_idx]
    # best_count = unordered[max_idx]
    
    # Let's do it vectorized to be fast.
    Ns = np.arange(2, max_N + 1)
    ordered = conv[2:max_N + 1]
    
    # Check if N//2 is powerful
    mids = Ns // 2
    # f is 1.0 for powerful numbers
    is_mid_powerful = f[mids] > 0.5
    
    # Only apply correction if N is even
    evens = (Ns % 2 == 0)
    correction = np.where(evens, is_mid_powerful.astype(np.int64), 0)
    
    # Integer division
    # Note: conv values are floats, close to integers.
    # Round them first.
    ordered_int = np.round(ordered).astype(np.int64)
    unordered = (ordered_int + correction) // 2
    
    max_idx = np.argmax(unordered)
    best_N = int(Ns[max_idx])
    best_count = int(unordered[max_idx])
    
    if time.time() > deadline - 30.0:
        # Quick verify and return
        pairs = []
        for a in powerful:
            b = best_N - a
            if b >= a and b in powerful_set:
                pairs.append([a, b])
        return {'N': best_N, 'pairs': pairs}
    
    # Verify best_N by explicit counting to be sure
    # And collect pairs
    pairs = []
    for a in powerful:
        b = best_N - a
        if b >= a and b in powerful_set:
            pairs.append([a, b])
            
    # The score is the number of pairs
    # We should ensure we didn't miss a better N due to FFT errors?
    # FFT on 0/1 inputs is usually exact enough for integers < 10^6 count.
    # Max count is ~ sqrt(N) ~ 10^4. Error is ~ 10^-10 * 10^8 = 10^-2. Safe.
    
    return {'N': best_N, 'pairs': pairs}
