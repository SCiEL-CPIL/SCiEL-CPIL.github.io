"""Rewritten program: FFT-based convolution to find N with most powerful-number pairs."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from collections import defaultdict


def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using a^2 * b^3 form."""
    powerful = set()
    powerful.add(1)
    max_b = int(round(limit ** (1/3))) + 2

    # Precompute square-free status
    sq_free = [True] * (max_b + 1)
    if max_b >= 0:
        sq_free[0] = False
    if max_b >= 1:
        sq_free[1] = True
    for p in range(2, int(math.isqrt(max_b)) + 1):
        if sq_free[p]:
            p2 = p * p
            for m in range(p2, max_b + 1, p2):
                sq_free[m] = False

    for b in range(1, max_b + 1):
        if not sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = int(math.isqrt(limit // b3))
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return sorted(powerful)


def _next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def construct():
    deadline = time.time() + 140
    start_time = time.time()

    # Choose limit based on available time
    # FFT of size ~2^22 = 4M is fast (~0.5s), 2^23 ~ 8M (~2s), 2^24 ~ 16M (~8s)
    # We want the sum array to be large enough to catch high-count N values
    # N can be up to 2*limit, so FFT size needs to be >= 2*limit
    # Let's aim for limit = 2^21 = 2M, so N up to 4M, FFT size 2^22 = 4M
    # Or limit = 2^22 = 4M, N up to 8M, FFT size 2^23 = 8M
    # Let's try limit = 4M (2^22), FFT size 2^23

    limit = 4_000_000  # 2^22

    powerful = _generate_powerful(limit)
    K = len(powerful)

    if K < 2:
        return {'N': 2, 'pairs': [[1, 1]]}

    powerful_set = set(powerful)

    # Create indicator array for FFT convolution
    # conv[i] = number of pairs (a, b) in powerful with a + b = i
    # We need FFT size >= 2 * limit
    max_sum = 2 * limit
    fft_size = _next_power_of_2(max_sum + 1)

    # Build indicator array
    ind = np.zeros(fft_size, dtype=np.float64)
    for p in powerful:
        ind[p] = 1.0

    # FFT convolution: conv = IFFT(FFT(ind) * FFT(ind))
    fft_ind = np.fft.fft(ind)
    conv = np.real(np.fft.ifft(fft_ind * fft_ind))

    # conv[i] ≈ number of ordered pairs (a,b) with a+b=i
    # For a <= b, count is (conv[i] + (1 if i is even and i/2 in powerful else 0)) / 2
    # Actually: unordered pairs with a<=b:
    #   If a != b: each unordered pair {a,b} appears twice in ordered pairs
    #   If a == b: appears once
    # So unordered count = (conv[i] + delta_i) / 2 where delta_i = 1 if i/2 is in powerful and i even
    # But we want a <= b, so:
    #   count = (conv[i] + diag_i) / 2 where diag_i = 1 if i is even and i//2 in powerful_set

    # Find the N with maximum pair count
    # N must be >= 2 and <= max_sum
    # Also N must be achievable: N >= 2 (min is 1+1=2)

    # Compute pair counts for all N
    # For efficiency, only consider N >= 2
    min_N = 2
    max_N = max_sum

    # Get conv values
    conv_counts = conv[min_N:max_N + 1].astype(np.int64)

    # Compute diagonal correction: for even N, if N//2 is powerful, add 1
    # diag[i] = 1 if (i is even) and (i//2 in powerful_set)
    # We need this for each N in [min_N, max_N]
    # Let's compute it efficiently
    # For even N = 2k, k ranges from 1 to max_N//2
    # Check if k is in powerful_set

    # Build a boolean array for powerful up to max_N//2
    max_k = max_N // 2
    is_pow = np.zeros(max_k + 1, dtype=np.int8)
    for p in powerful:
        if p <= max_k:
            is_pow[p] = 1

    # For N in [min_N, max_N], N is at index N - min_N in conv_counts
    # N is even iff (N - min_N) has same parity as min_N
    # min_N = 2, so N even iff (N-2) even iff N even
    # N//2 = (N)//2
    # Index for k = N//2 in is_pow is N//2

    # Vectorized: for each position idx (0-based) in conv_counts, N = idx + min_N
    idxs = np.arange(max_N - min_N + 1, dtype=np.int64)
    N_vals = idxs + min_N
    even_mask = (N_vals % 2 == 0)
    k_vals = N_vals // 2
    # k_vals ranges from min_N//2=1 to max_N//2
    diag = np.zeros(len(N_vals), dtype=np.int64)
    diag[even_mask] = is_pow[k_vals[even_mask]]

    pair_counts = (conv_counts + diag) // 2

    # Find the maximum
    best_idx = int(np.argmax(pair_counts))
    best_N = int(N_vals[best_idx])
    best_count = int(pair_counts[best_idx])

    if best_count < 1:
        return {'N': 2, 'pairs': [[1, 1]]}

    # Generate the actual pairs for best_N
    pairs = []
    for a in powerful:
        if a > best_N // 2:
            break
        b = best_N - a
        if b >= a and b in powerful_set:
            pairs.append([a, b])

    # Verify count
    actual_count = len(pairs)
    if actual_count != best_count:
        # FFT rounding error; use actual count
        best_count = actual_count

    if not pairs:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}

    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
