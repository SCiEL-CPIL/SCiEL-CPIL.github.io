import time
import numpy as np
from math import isqrt

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using vectorized NumPy.
    Powerful numbers are of the form a^2 * b^3 where b is square-free.
    """
    max_b = int(limit ** (1.0 / 3.0)) + 3
    # Sieve for square-free numbers up to max_b
    is_sq_free = np.ones(max_b + 1, dtype=bool)
    is_sq_free[0] = False
    for i in range(2, isqrt(max_b) + 1):
        sq = i * i
        is_sq_free[sq::sq] = False
    is_sq_free[1] = True

    sq_free_bs = np.where(is_sq_free)[0]
    # Filter to b^3 <= limit
    valid_mask = sq_free_bs.astype(np.int64) ** 3 <= limit
    sq_free_bs = sq_free_bs[valid_mask]

    all_powerful = []
    for b in sq_free_bs:
        b3 = int(b) ** 3
        max_a = isqrt(limit // b3)
        # Vectorized: a^2 * b3 for a = 1..max_a
        a_vals = np.arange(1, max_a + 1, dtype=np.int64)
        vals = a_vals * a_vals * b3
        vals = vals[vals <= limit]
        all_powerful.append(vals)

    if all_powerful:
        arr = np.concatenate(all_powerful)
        arr = np.unique(arr)
    else:
        arr = np.array([], dtype=np.int64)

    return arr  # return sorted numpy array

def _count_pairs(N, powerful_arr, powerful_set):
    """Count unordered pairs (a,b) with a<=b, a+b=N, both powerful.
    powerful_arr is sorted numpy array; powerful_set is a set for O(1) lookup.
    """
    # Binary search for elements <= N//2
    half = N // 2
    idx_end = np.searchsorted(powerful_arr, half, side='right')
    candidates = powerful_arr[:idx_end]
    b_vals = N - candidates
    # Check which b_vals are in powerful_set
    mask = np.array([bv in powerful_set for bv in b_vals])
    count = int(np.sum(mask))
    pairs = [[int(candidates[i]), int(b_vals[i])] for i in range(len(candidates)) if mask[i]]
    return count, pairs

def construct():
    t0 = time.time()
    deadline = t0 + 118

    best_result = {'N': 2, 'pairs': [[1, 1]]}
    best_count = 1

    # Try progressively larger limits, vectorized generation
    limits_to_try = [50_000_000, 150_000_000, 300_000_000]

    for limit in limits_to_try:
        if time.time() > deadline - 40:
            break

        powerful_arr = _generate_powerful(limit)
        n_powerful = len(powerful_arr)
        if n_powerful < 10:
            continue

        powerful_set = set(powerful_arr.tolist())

        # FFT convolution to find top N candidates
        size = 1
        while size < 2 * limit + 1:
            size *= 2

        # For very large sizes, FFT may be too slow; cap the effective limit for FFT
        fft_limit = min(limit, 100_000_000)
        if fft_limit < limit:
            # Use a subset for FFT: only powerful numbers <= fft_limit
            mask = powerful_arr <= fft_limit
            pw_subset = powerful_arr[mask]
        else:
            pw_subset = powerful_arr

        fft_size = 1
        while fft_size < 2 * fft_limit + 1:
            fft_size *= 2

        indicator = np.zeros(fft_size, dtype=np.float64)
        indicator[pw_subset.astype(np.int64)] = 1.0

        fft_ind = np.fft.fft(indicator)
        conv = np.fft.ifft(fft_ind * fft_ind)
        conv_real = np.round(np.real(conv)).astype(np.int64)

        # Find top K candidates by convolution value
        max_N_candidate = 2 * fft_limit
        top_k = min(500, max_N_candidate - 1)
        candidate_indices = np.argsort(conv_real[2:max_N_candidate + 1])[-top_k:][::-1] + 2
        candidates = [int(idx) for idx in candidate_indices]

        # Also add candidates from the full range using sums of small powerful numbers
        small_pw = powerful_arr[:min(3000, n_powerful)]
        seen = set(candidates)
        extra_candidates = []
        for i in range(min(1500, len(small_pw))):
            for j in range(i, min(i + 100, len(small_pw))):
                N_cand = int(small_pw[i] + small_pw[j])
                if 2 <= N_cand <= 2 * limit and N_cand not in seen:
                    seen.add(N_cand)
                    extra_candidates.append(N_cand)
                if len(extra_candidates) >= 300:
                    break
            if len(extra_candidates) >= 300:
                break
        candidates.extend(extra_candidates)

        for N_candidate in candidates:
            if time.time() > deadline - 15:
                break

            count, pairs = _count_pairs(N_candidate, powerful_arr, powerful_set)

            if count > best_count:
                best_count = count
                best_result = {'N': N_candidate, 'pairs': pairs}

        # Random sampling
        n_random = 30
        for _ in range(n_random):
            if time.time() > deadline - 10:
                break
            N_candidate = int(np.random.randint(2, 2 * limit + 1))
            count, pairs = _count_pairs(N_candidate, powerful_arr, powerful_set)
            if count > best_count:
                best_count = count
                best_result = {'N': N_candidate, 'pairs': pairs}

    return best_result
