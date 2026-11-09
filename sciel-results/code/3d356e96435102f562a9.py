"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using recursive prime-based generation."""
    max_p = isqrt(limit)
    if max_p < 2:
        return [1]
    is_prime = [True] * (max_p + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(max_p) + 1):
        if is_prime[i]:
            for j in range(i * i, max_p + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, max_p + 1) if is_prime[i]]
    
    result = set([1])
    
    def gen(idx, current):
        if current > limit:
            return
        if idx >= len(primes):
            if current > 1:
                result.add(current)
            return
        p = primes[idx]
        gen(idx + 1, current)
        val = current * p * p
        while val <= limit:
            result.add(val)
            gen(idx + 1, val)
            val *= p
    
    gen(0, 1)
    return sorted(result)

def construct():
    start_time = time.time()
    deadline = start_time + 125
    
    L = 30_000_000
    powerful = _generate_powerful(L)
    n_pow = len(powerful)
    pow_set = set(powerful)
    
    if time.time() > deadline - 60:
        L = 10_000_000
        powerful = _generate_powerful(L)
        n_pow = len(powerful)
        pow_set = set(powerful)
    
    # FFT convolution to identify top candidate sums
    size = 2 * L + 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    
    ind = np.zeros(fft_size, dtype=np.float64)
    for a in powerful:
        ind[a] = 1.0
    
    f = np.fft.rfft(ind)
    f_sq = f * f
    conv = np.fft.irfft(f_sq, n=fft_size)
    
    max_idx = min(2 * L + 1, fft_size)
    conv_relevant = conv[:max_idx]
    
    top_indices = np.argsort(conv_relevant)[::-1][:100]
    
    best_N = None
    best_count = 0
    best_pairs = []
    
    for idx in top_indices:
        N = int(idx)
        if N < 2 or N > 10**18:
            continue
        
        pairs = []
        for a in powerful:
            b = N - a
            if b < a:
                break
            if b in pow_set:
                pairs.append([a, b])
        
        count = len(pairs)
        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = pairs
        
        if time.time() > deadline - 5:
            break
    
    if best_N is None:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
