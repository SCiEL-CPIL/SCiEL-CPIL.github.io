import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Strategy: Generate powerful numbers up to a limit L, use FFT convolution
    # to find N with maximum representations as sum of two powerful numbers.
    
    # Powerful number: every prime factor appears with exponent >= 2.
    # Generation: n = a^2 * b^3 where we iterate over all a,b.
    # This overcounts but a set handles it.
    
    L = 2 * 10**7
    
    powerful_set = set()
    powerful_set.add(1)
    
    # Generate powerful numbers via n = a^2 * b^3
    max_b = int(round(L ** (1/3))) + 2
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
    
    if time.time() > deadline - 100.0:
        # Fallback with smaller L
        L = 5 * 10**6
        powerful_set = {1}
        max_b = int(round(L ** (1/3))) + 2
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
    
    # FFT convolution to find best N in [2, 2*L]
    fft_size = 1
    while fft_size < 2 * L + 1:
        fft_size *= 2
    
    f = np.zeros(fft_size, dtype=np.float64)
    for p in powerful:
        if p < fft_size:
            f[p] = 1.0
    
    f_fft = np.fft.fft(f)
    conv_fft = f_fft * f_fft
    conv = np.fft.ifft(conv_fft).real
    
    # For each N, count pairs (a,b) with a <= b, a+b=N, both powerful.
    # conv[N] = #{(a,b): a+b=N, a,b in powerful} (ordered pairs including a=b)
    # unordered count = (conv[N] + (1 if N even and N/2 in powerful else 0)) // 2
    
    max_N_search = min(2 * L, fft_size - 1)
    Ns = np.arange(2, max_N_search + 1)
    ordered = conv[2:max_N_search + 1]
    mids = Ns // 2
    is_mid_powerful = f[mids] > 0.5
    evens = (Ns % 2 == 0)
    correction = np.where(evens, is_mid_powerful.astype(np.int64), 0)
    ordered_int = np.round(ordered).astype(np.int64)
    unordered = (ordered_int + correction) // 2
    
    max_idx = np.argmax(unordered)
    best_N = int(Ns[max_idx])
    best_count = int(unordered[max_idx])
    
    # Collect pairs for best_N
    pairs_best = []
    for a in powerful:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set:
            pairs_best.append([a, b])
    
    best_overall_N = best_N
    best_overall_count = len(pairs_best)
    best_overall_pairs = pairs_best
    
    if time.time() > deadline - 20.0:
        return {'N': best_overall_N, 'pairs': best_overall_pairs}
    
    # Try to improve by looking at top candidates from FFT
    # Get top K N values and verify/count them directly
    top_k = 50
    top_indices = np.argsort(unordered)[::-1][:top_k]
    
    for idx in top_indices:
        if time.time() > deadline - 10.0:
            break
        N = int(Ns[idx])
        if N < 2 or N > 10**18:
            continue
        # Direct count
        count = 0
        pairs = []
        for a in powerful:
            if a > N // 2:
                break
            b = N - a
            if b in powerful_set:
                count += 1
                pairs.append([a, b])
        if count > best_overall_count:
            best_overall_count = count
            best_overall_N = N
            best_overall_pairs = pairs
    
    if best_overall_pairs is None:
        best_overall_pairs = []
    
    return {'N': best_overall_N, 'pairs': best_overall_pairs}
