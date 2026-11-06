import time
import sys
import numpy as np
from math import isqrt

def _generate_powerful_numbers(limit):
    """
    Generate all powerful numbers up to limit.
    A powerful number is of the form u^2 * v^3 where v is square-free.
    We iterate v from 1 to limit^(1/3).
    For each v, we iterate u from 1 to sqrt(limit/v^3).
    """
    powerful = set()
    # Add 1 as a powerful number (1 = 1^2 * 1^3)
    powerful.add(1)
    
    # v must satisfy v^3 <= limit
    max_v = int(round(limit ** (1.0 / 3.0))) + 2
    if max_v < 1:
        max_v = 1

    for v in range(1, max_v + 1):
        v3 = v * v * v
        if v3 > limit:
            break
        
        # u^2 * v3 <= limit => u^2 <= limit / v3
        max_u = isqrt(limit // v3)
        for u in range(1, max_u + 1):
            val = u * u * v3
            if val <= limit:
                powerful.add(val)
            else:
                break
                
    return sorted(powerful)

def construct():
    start_time = time.time()
    # Deadline guard: 150s limit, keep 15s safety margin
    deadline = start_time + 135.0

    best_result = {'N': 5, 'pairs': [[1, 4]]}
    best_count = 1

    # Try larger limits first to maximize score.
    # M=180M requires ~1.5GB (float32) or ~3GB (float64) for FFT array.
    # Let's try 180M with float32 first. If it fails, drop down.
    limits_to_try = [180_000_000, 100_000_000, 50_000_000, 20_000_000]
    
    for M in limits_to_try:
        try:
            if time.time() > deadline - 20:
                break
                
            gen_start = time.time()
            powerful_list = _generate_powerful_numbers(M)
            gen_time = time.time() - gen_start
            
            if time.time() > deadline - 15:
                continue
                
            K = len(powerful_list)
            if K == 0:
                continue
                
            # Prepare FFT array
            # Size S = 2*M + 1 to avoid circular convolution overlap for sums up to 2*M
            S = 2 * M + 1
            
            # Use float32 to save memory, allowing larger M
            # Precision of float32 is ~7 decimal digits. Max count is K ~ 2*sqrt(M).
            # For M=180M, K ~ 2*13400 = 26800. This is well within float32 precision.
            arr = np.zeros(S, dtype=np.float32)
            arr[powerful_list] = 1.0
            
            # Perform FFT
            fft_start = time.time()
            fft_arr = np.fft.rfft(arr)
            conv = np.fft.irfft(fft_arr * fft_arr, n=S)
            fft_time = time.time() - fft_start
            
            if time.time() > deadline - 10:
                continue
                
            # Find the index with the maximum number of representations
            # conv[i] is the number of ordered pairs (a,b) such that a+b=i.
            # We want to maximize the number of unordered pairs a<=b.
            # Max conv[i] is a good proxy.
            
            # argmax is O(N)
            best_idx = int(np.argmax(conv))
            N_cand = best_idx
            
            # Verify and extract pairs for this N
            # We need distinct pairs a <= b such that a+b=N and a,b are powerful.
            powerful_set = set(powerful_list)
            pairs = []
            
            # Iterate through powerful numbers a
            # To ensure a <= b, we only need to check a <= N/2
            half_N = N_cand // 2
            
            # powerful_list is sorted. We can break early.
            for a in powerful_list:
                if a > half_N:
                    break
                b = N_cand - a
                if b in powerful_set:
                    pairs.append([a, b])
            
            current_count = len(pairs)
            
            if current_count > best_count:
                best_count = current_count
                best_result = {'N': N_cand, 'pairs': pairs}
                
        except MemoryError:
            # If M is too large, try next smaller limit
            continue
        except Exception as e:
            # If any other error occurs, try next smaller limit
            continue
            
    return best_result

if __name__ == "__main__":
    res = construct()
    print(res['N'], len(res['pairs']))
