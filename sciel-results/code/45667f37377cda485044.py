"""FFT-based powerful-number pair counting with exact extraction."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
import numpy as np

def _generate_powerfuls(limit):
    """Generate all powerful numbers <= limit.
    A powerful number has the form a^2 * b^3 where b is square-free.
    """
    powerfuls = set()
    max_b = int(limit ** (1/3)) + 2
    sqrt_max_b = isqrt(max_b)
    primes = []
    for x in range(2, sqrt_max_b + 1):
        is_p = True
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(x)
    
    for b in range(1, max_b + 1):
        is_sq_free = True
        for p in primes:
            if p * p > b:
                break
            if b % (p * p) == 0:
                is_sq_free = False
                break
        if not is_sq_free:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerfuls.add(val)
    return powerfuls

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill
    
    # Use FFT with L = 2*10^7
    # FFT size: next power of 2 >= 2*L = 4*10^7 -> 2^26 = 67108864
    # Memory: 67M * 16 bytes = ~1.07 GB for complex128 -- should be OK
    # Time: FFT ~ 1-3 seconds
    L = 2 * 10**7
    
    powerfuls = _generate_powerfuls(L)
    powerfuls_sorted = sorted(powerfuls)
    powerfuls_set = set(powerfuls)
    n = len(powerfuls_sorted)
    
    if time.time() > deadline - 30:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Build indicator array
    max_sum = 2 * L
    # FFT size: next power of 2 >= max_sum + 1
    fft_size = 1
    while fft_size < max_sum + 1:
        fft_size *= 2
    
    # Create indicator: 1 at positions where powerful number exists
    indicator = np.zeros(fft_size, dtype=np.float64)
    for p in powerfuls_sorted:
        indicator[p] = 1.0
    
    # Convolution: indicator * indicator gives count of ordered pairs (a,b) with a+b=N
    # We want unordered with a <= b, so count_unordered = (convolution[N] + (1 if N is even and N/2 is powerful))/2
    # Actually: convolution[N] = #{(a,b): a+b=N, a,b powerful} = #{a<=b} + #{a>b}
    # By symmetry: #{a>b} = #{a<b}
    # So convolution[N] = 2*#{a<b} + #{a=b}
    # #{a=b} = 1 if N is even and N/2 is powerful, else 0
    # #{a<=b} = #{a<b} + #{a=b} = (convolution[N] - #{a=b})/2 + #{a=b} = (convolution[N] + #{a=b})/2
    
    t0 = time.time()
    fft_indicator = np.fft.fft(indicator)
    conv = np.fft.ifft(fft_indicator * fft_indicator).real
    conv = np.round(conv).astype(np.int64)
    fft_time = time.time() - t0
    # print(f"FFT took {fft_time:.2f}s, size={fft_size}")
    
    if time.time() > deadline - 20:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Find the N that maximizes the number of unordered pairs a<=b with a+b=N
    # For each N, count = (conv[N] + diag[N]) / 2 where diag[N]=1 if N even and N/2 powerful
    # We want to maximize this count.
    
    # Efficiently: compute score[N] = conv[N] + diag_indicator[N]
    # Then the number of unordered pairs = score[N] // 2
    
    # diag_indicator: 1 at N where N is even and N/2 is powerful
    diag = np.zeros(fft_size, dtype=np.int64)
    for p in powerfuls_sorted:
        N_val = 2 * p
        if N_val < fft_size:
            diag[N_val] = 1
    
    score = conv + diag
    # Number of unordered pairs = score // 2 (since score is always even: conv[N] + diag[N] is even)
    
    # Find max score in range [2, 2*L]
    valid_range = slice(2, 2 * L + 1)
    max_idx = 2 + np.argmax(score[valid_range])
    best_N = int(max_idx)
    best_score = int(score[best_N])
    best_count = best_score // 2
    
    if time.time() > deadline - 10:
        # Extract pairs quickly
        pairs = []
        for a in powerfuls_sorted:
            if a > best_N // 2:
                break
            b = best_N - a
            if b in powerfuls_set:
                pairs.append([a, b])
        return {'N': best_N, 'pairs': pairs}
    
    # Extract pairs for best_N
    pairs = []
    for a in powerfuls_sorted:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerfuls_set:
            pairs.append([a, b])
    
    best_final_N = best_N
    best_final_pairs = pairs
    best_final_count = len(pairs)
    
    # Check top 30 candidates to ensure we have the true max
    # Get top 30 indices by score
    top_k = 30
    # We only care about range [2, 2*L]
    scores_valid = score[2:2*L+1]
    top_indices = np.argsort(scores_valid)[::-1][:top_k]
    
    for idx in top_indices:
        cand_N = int(idx + 2)
        cand_pairs = []
        for a in powerfuls_sorted:
            if a > cand_N // 2:
                break
            b = cand_N - a
            if b in powerfuls_set:
                cand_pairs.append([a, b])
        if len(cand_pairs) > best_final_count:
            best_final_count = len(cand_pairs)
            best_final_N = cand_N
            best_final_pairs = cand_pairs
    
    if time.time() > deadline - 5:
        return {'N': best_final_N, 'pairs': best_final_pairs}
    
    return {'N': best_final_N, 'pairs': best_final_pairs}

# EVOLVE-BLOCK-END
