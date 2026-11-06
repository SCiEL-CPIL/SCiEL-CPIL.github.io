"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit.
    Powerful = a^2 * b^3 where b is square-free.
    Returns a sorted list and a set.
    """
    max_b = int(limit ** (1.0 / 3.0)) + 2
    # Sieve for square-free numbers up to max_b
    is_sq_free = [True] * (max_b + 1)
    is_sq_free[0] = False
    for i in range(2, isqrt(max_b) + 1):
        sq = i * i
        for j in range(sq, max_b + 1, sq):
            is_sq_free[j] = False
    is_sq_free[1] = True

    powerful_set = set()
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
                powerful_set.add(val)
    powerful_list = sorted(powerful_set)
    return powerful_list, powerful_set

def construct():
    t0 = time.time()
    deadline = t0 + 135  # 15s safety margin before 150s kill
    
    best_N = 2
    best_pairs = [[1, 1]]
    best_count = 1
    
    # Strategy: Use FFT convolution to count representations of each N as sum of two powerful numbers.
    # For a limit L, we generate powerful numbers up to L, create a boolean array of size 2L+1,
    # and use FFT to compute the convolution. The peak of the convolution gives the N with max representations.
    # We try progressively larger limits until time runs out.
    
    import numpy as np
    
    # Try a range of limits. The key is that larger L gives more powerful numbers and potentially more collisions.
    # But FFT memory is O(L log L), so we need to be careful.
    # For L = 1e7, array size ~2e7, FFT of size ~4e7 -> ~320MB for complex, feasible.
    # For L = 5e7, array size ~1e8, FFT of size ~2e8 -> ~1.6GB, might be tight.
    
    limits_to_try = [5_000_000, 10_000_000, 20_000_000, 40_000_000, 80_000_000]
    
    for limit in limits_to_try:
        if time.time() > deadline - 30:
            break
        
        try:
            powerful_list, powerful_set = _generate_powerful(limit)
            n_pow = len(powerful_list)
            
            if n_pow < 10:
                continue
            
            # Create boolean array for FFT convolution
            # Array size must be at least 2*limit + 1 to capture all sums up to 2*limit
            # Use next power of 2 for FFT efficiency
            arr_size = 2 * limit + 1
            # Round up to next power of 2
            fft_size = 1
            while fft_size < arr_size:
                fft_size <<= 1
            
            # Check if we have enough memory (rough estimate: fft_size * 16 bytes for complex128)
            mem_estimate = fft_size * 16
            if mem_estimate > 2_000_000_000:  # 2GB limit
                # Fallback to direct counting for this limit
                counts = defaultdict(int)
                for i in range(n_pow):
                    a = powerful_list[i]
                    for j in range(i, n_pow):
                        b = powerful_list[j]
                        counts[a + b] += 1
                    if time.time() > deadline - 15:
                        break
                if counts:
                    candidate_N = max(counts, key=counts.get)
                    candidate_count = counts[candidate_N]
                    if candidate_count > best_count:
                        best_count = candidate_count
                        best_N = candidate_N
                        pairs = []
                        for a in powerful_list:
                            if 2 * a > candidate_N:
                                break
                            b = candidate_N - a
                            if b in powerful_set and a <= b:
                                pairs.append([a, b])
                        best_pairs = pairs
                continue
            
            # Create the signal
            signal = np.zeros(fft_size, dtype=np.float64)
            for p in powerful_list:
                if p < fft_size:
                    signal[p] = 1.0
            
            # FFT convolution
            fft_sig = np.fft.fft(signal)
            fft_conv = fft_sig * fft_sig
            result = np.fft.ifft(fft_conv).real
            
            # The result[i] gives the number of pairs (a,b) with a+b=i
            # But this counts ordered pairs (a,b) and (b,a) separately, and (a,a) once.
            # Actually, convolution of f*f gives sum over k of f[k]*f[i-k], which counts
            # all ordered pairs (k, i-k) where both are in the set.
            # For a != b, the pair (a,b) and (b,a) are both counted, so we need to divide by 2 for those.
            # For a == b, it's counted once.
            # So: number of unordered pairs with a <= b = (result[i] + count_of_a_eq_b_for_i) / 2
            # where count_of_a_eq_b_for_i is 1 if i/2 is powerful and i is even, else 0.
            
            # Find the peak
            # We only care about indices up to 2*limit
            max_idx = min(2 * limit + 1, fft_size)
            candidate_region = result[:max_idx]
            
            # Compute the actual number of unordered pairs for each N
            # For efficiency, let's first find the top few candidates from the raw convolution
            # and then compute exact counts for those.
            
            # Get top 50 candidates by raw convolution value
            top_indices = np.argsort(candidate_region)[-50:][::-1]
            
            for idx in top_indices:
                idx = int(idx)
                if idx < 2:
                    continue
                
                # Count exact number of unordered pairs (a,b) with a <= b, a+b=idx, a,b powerful
                count = 0
                for a in powerful_list:
                    if a > idx // 2:
                        break
                    b = idx - a
                    if b in powerful_set:
                        count += 1
                
                if count > best_count:
                    best_count = count
                    best_N = idx
                    pairs = []
                    for a in powerful_list:
                        if a > idx // 2:
                            break
                        b = idx - a
                        if b in powerful_set:
                            pairs.append([a, b])
                    best_pairs = pairs
                
                if time.time() > deadline - 15:
                    break
            
        except Exception as e:
            # If FFT fails (e.g., memory), fall back to direct counting for smaller limit
            if time.time() > deadline - 20:
                break
            # Try a smaller limit with direct counting
            small_limit = max(1_000_000, limit // 10)
            try:
                powerful_list, powerful_set = _generate_powerful(small_limit)
                n_pow = len(powerful_list)
                counts = defaultdict(int)
                for i in range(n_pow):
                    a = powerful_list[i]
                    for j in range(i, n_pow):
                        b = powerful_list[j]
                        counts[a + b] += 1
                    if time.time() > deadline - 15:
                        break
                if counts:
                    candidate_N = max(counts, key=counts.get)
                    candidate_count = counts[candidate_N]
                    if candidate_count > best_count:
                        best_count = candidate_count
                        best_N = candidate_N
                        pairs = []
                        for a in powerful_list:
                            if 2 * a > candidate_N:
                                break
                            b = candidate_N - a
                            if b in powerful_set and a <= b:
                                pairs.append([a, b])
                        best_pairs = pairs
            except:
                pass
    
    # Ensure best_pairs is consistent with best_N
    if best_pairs:
        # Verify and regenerate if needed
        final_pairs = []
        for a, b in best_pairs:
            if a + b == best_N and a <= b:
                final_pairs.append([a, b])
        best_pairs = final_pairs
    else:
        best_pairs = [[1, 1]]
        best_N = 2
    
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
