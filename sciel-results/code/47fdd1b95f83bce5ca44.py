import time
import numpy as np
from math import isqrt
from bisect import bisect_left, bisect_right

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit.
    Powerful = a^2 * b^3 where b is square-free.
    """
    max_b = int(limit ** (1.0 / 3.0)) + 2
    is_sq_free = np.ones(max_b + 1, dtype=bool)
    is_sq_free[0] = False
    for i in range(2, isqrt(max_b) + 1):
        sq = i * i
        is_sq_free[sq::sq] = False
    is_sq_free[1] = True

    powerful = set()
    for b in range(1, max_b + 1):
        if not is_sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return powerful

def construct():
    t0 = time.time()
    deadline = t0 + 115

    best_result = None
    best_count = 0

    # Strategy: Use FFT convolution to efficiently find N with many representations.
    # conv[N] = number of ordered pairs (a,b) of powerful numbers with a+b=N.
    # This is O(L log L) instead of O(n^2) for brute force pairwise sums.
    
    limits_to_try = [5_000_000, 10_000_000, 20_000_000]
    
    for limit in limits_to_try:
        if time.time() > deadline - 25:
            break

        powerful = _generate_powerful(limit)
        n_powerful = len(powerful)
        
        if n_powerful < 10:
            continue

        powerful_list = sorted(powerful)
        powerful_set = powerful

        # FFT size: need at least 2*limit+1, round up to power of 2
        size = 1
        while size < 2 * limit + 1:
            size *= 2
        
        indicator = np.zeros(size, dtype=np.float64)
        for p in powerful_list:
            indicator[p] = 1.0
        
        # FFT convolution
        fft_ind = np.fft.fft(indicator)
        conv = np.fft.ifft(fft_ind * fft_ind)
        conv_real = np.round(np.real(conv)).astype(np.int64)
        
        # Find top N candidates by conv_real value
        max_N_candidate = 2 * limit
        top_k = min(200, max_N_candidate - 1)
        candidate_indices = np.argsort(conv_real[2:max_N_candidate + 1])[-top_k:][::-1] + 2
        
        for idx in candidate_indices:
            if time.time() > deadline - 10:
                break
            N_candidate = int(idx)
            
            # Verify and count actual unordered pairs with a <= b
            count = 0
            pairs = []
            for a in powerful_list:
                if a > N_candidate // 2:
                    break
                b = N_candidate - a
                if b in powerful_set:
                    count += 1
                    pairs.append([a, b])
            
            if count > best_count:
                best_count = count
                best_result = {'N': N_candidate, 'pairs': pairs}
        
        # Check a few random N as backup in case FFT has numerical issues
        n_random = 30
        for _ in range(n_random):
            if time.time() > deadline - 5:
                break
            N_candidate = int(np.random.randint(2, 2 * limit + 1))
            count = 0
            pairs = []
            for a in powerful_list:
                if a > N_candidate // 2:
                    break
                b = N_candidate - a
                if b in powerful_set:
                    count += 1
                    pairs.append([a, b])
            if count > best_count:
                best_count = count
                best_result = {'N': N_candidate, 'pairs': pairs}

    if best_result is None:
        limit = 100_000
        powerful = _generate_powerful(limit)
        powerful_list = sorted(powerful)
        powerful_set = powerful
        best_result = {'N': 2, 'pairs': [[1, 1]]}

    return best_result
