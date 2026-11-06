# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    # Powerful: a^2 * b^3 where b is square-free
    max_b = int(round(limit ** (1.0 / 3.0))) + 2
    # Sieve for square-free
    is_sq_free = np.ones(max_b + 1, dtype=bool)
    for i in range(2, isqrt(max_b) + 1):
        sq = i * i
        is_sq_free[sq::sq] = False
    is_sq_free[0] = False
    is_sq_free[1] = True  # 1 is square-free
    
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
    deadline = t0 + 110  # 110s safety margin
    
    # Try multiple limits, starting small, going larger if time permits
    best_result = None
    best_count = 0
    
    for limit in [2_000_000, 5_000_000, 10_000_000, 20_000_000]:
        if time.time() > deadline - 30:
            break
        
        # Generate powerful numbers up to limit
        powerful = _generate_powerful(limit)
        n_powerful = len(powerful)
        
        if n_powerful < 2:
            continue
        
        # Create indicator array
        arr = np.zeros(limit + 1, dtype=np.float64)
        for p in powerful:
            arr[p] = 1.0
        
        # FFT-based autocorrelation: convolve arr with reversed arr
        # Use next power of 2 for FFT size
        fft_size = 1
        while fft_size < 2 * (limit + 1):
            fft_size *= 2
        
        F = np.fft.fft(arr, n=fft_size)
        # Autocorrelation: ifft(F * conj(F)) gives circular autocorrelation
        # We want linear autocorrelation, so use zero-padding (already done via fft_size)
        corr = np.fft.ifft(F * np.conj(F)).real
        # corr[k] = sum_i arr[i] * arr[i+k] for k >= 0
        # But we want sum of arr[i]*arr[j] where i+j = N, i.e., convolution
        # Actually, autocorrelation gives sum_i arr[i]*arr[i+k]
        # For sums: we want (arr * arr)[N] = sum_i arr[i] * arr[N-i]
        # That's the linear convolution, not autocorrelation.
        # Let's do proper convolution: ifft(F * F)
        conv = np.fft.ifft(F * F).real
        # conv[N] = sum_{i=0}^{N} arr[i] * arr[N-i]
        # This counts ordered pairs (a, b) with a + b = N
        # We want unordered pairs with a <= b, so divide by 2 and handle a==b
        
        # conv[N] counts (a,b) and (b,a) separately, plus (a,a) once
        # Number of unordered pairs with a <= b: (conv[N] + count_of_a==b) / 2
        # where count_of_a==b is 1 if N/2 is powerful and N is even, else 0
        
        # For scoring, the task says "distinct pairs a <= b", so we need unordered
        # conv[N] = 2 * (number of pairs with a < b) + (1 if N/2 is powerful and N even)
        # number of pairs a <= b = (conv[N] + (1 if N%2==0 and arr[N//2]>0 else 0)) // 2
        
        # Find N with max count
        # conv is valid for N from 0 to 2*limit
        # We need N >= 2
        max_idx = np.argmax(conv[2:]) + 2
        N_best = int(max_idx)
        conv_val = int(round(conv[N_best]))
        
        # Compute actual number of unordered pairs
        if N_best % 2 == 0 and N_best // 2 <= limit and arr[N_best // 2] > 0:
            n_pairs = (conv_val + 1) // 2
        else:
            n_pairs = conv_val // 2
        
        if n_pairs > best_count:
            best_count = n_pairs
            # Generate the actual pairs
            powerful_list = sorted(powerful)
            powerful_set = powerful
            pairs = []
            for a in powerful_list:
                if 2 * a > N_best:
                    break
                b = N_best - a
                if b in powerful_set:
                    pairs.append([a, b])
            best_result = {'N': N_best, 'pairs': pairs}
        
        if time.time() > deadline - 20:
            break
    
    if best_result is None:
        # Fallback: small search
        powerful = _generate_powerful(100000)
        arr = np.zeros(100001, dtype=np.float64)
        for p in powerful:
            arr[p] = 1.0
        fft_size = 1
        while fft_size < 200002:
            fft_size *= 2
        F = np.fft.fft(arr, n=fft_size)
        conv = np.fft.ifft(F * F).real
        max_idx = int(np.argmax(conv[2:]) + 2)
        N_best = max_idx
        conv_val = int(round(conv[N_best]))
        if N_best % 2 == 0 and N_best // 2 <= 100000 and arr[N_best // 2] > 0:
            n_pairs = (conv_val + 1) // 2
        else:
            n_pairs = conv_val // 2
        powerful_set = powerful
        pairs = []
        for a in sorted(powerful_set):
            if 2 * a > N_best:
                break
            b = N_best - a
            if b in powerful_set:
                pairs.append([a, b])
        best_result = {'N': N_best, 'pairs': pairs}
    
    return best_result
# EVOLVE-BLOCK-END
