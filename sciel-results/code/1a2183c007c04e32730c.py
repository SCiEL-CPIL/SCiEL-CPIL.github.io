import time
import sys
import numpy as np
from math import isqrt

def _generate_powerful_numbers(limit):
    """
    Generate all powerful numbers up to limit.
    A powerful number is of the form u^2 * v^3 where v is square-free.
    We generate all u^2 * v^3 and use a set to handle duplicates.
    """
    powerful = set()
    max_v = int(limit ** (1.0 / 3.0)) + 2
    
    for v in range(1, max_v + 1):
        v3 = v * v * v
        if v3 > limit:
            break
        max_u = isqrt(limit // v3)
        for u in range(1, max_u + 1):
            val = u * u * v3
            if val <= limit:
                powerful.add(val)
            else:
                break
    return sorted(powerful)

def _count_pairs_for_N(N, powerful_list, powerful_set):
    """
    Count pairs (a, b) with a <= b, a + b = N, both powerful.
    Returns (count, pairs).
    """
    count = 0
    pairs = []
    half_N = N // 2
    # powerful_list is sorted
    for a in powerful_list:
        if a > half_N:
            break
        b = N - a
        if b in powerful_set:
            count += 1
            pairs.append([a, b])
    return count, pairs

def construct():
    start_time = time.time()
    # Deadline guard: 150s limit, keep 10s safety margin
    deadline = start_time + 140.0

    best_result = {'N': 5, 'pairs': [[1, 4]]}
    
    # Try multiple M values, starting from larger to find larger N with potentially more representations
    # M is the max powerful number generated. N can be up to 2*M.
    # Memory for FFT of size 2*M+1:
    # float32 input: (2*M+1)*4 bytes
    # complex128 FFT: (M+1)*16 bytes
    # Total ~ 24*M bytes.
    # For M=100M, 2.4GB. Might be tight.
    # For M=80M, 1.9GB. Safer.
    # For M=50M, 1.2GB. Very safe.
    
    limits_to_try = [100_000_000, 80_000_000, 50_000_000, 30_000_000]
    
    for M in limits_to_try:
        try:
            if time.time() > deadline - 40:
                break
                
            # 1. Generate powerful numbers
            gen_start = time.time()
            powerful_list = _generate_powerful_numbers(M)
            gen_time = time.time() - gen_start
            
            if time.time() > deadline - 25:
                continue
                
            K = len(powerful_list)
            if K == 0:
                continue
            
            # 2. FFT Convolution
            # We want to count pairs (a,b) such that a+b=N, a,b in powerful_list.
            # This is the auto-correlation of the indicator function of powerful numbers.
            S = 2 * M + 1
            
            # Use float32 to save memory
            arr = np.zeros(S, dtype=np.float32)
            arr[powerful_list] = 1.0
            
            # Perform FFT
            # numpy.fft.rfft converts to complex128 internally
            fft_arr = np.fft.rfft(arr)
            conv = np.fft.irfft(fft_arr * fft_arr, n=S)
            
            # 3. Find best N
            # Get top 200 indices by conv value
            top_k = 200
            if S > top_k:
                top_indices = np.argpartition(conv, -top_k)[-top_k:]
            else:
                top_indices = np.arange(S)
                
            # Sort these top indices by conv value descending
            top_indices = top_indices[np.argsort(conv[top_indices])[::-1]]
            
            best_N = 2
            best_count = 1
            best_pairs = [[1, 1]]
            
            powerful_set = set(powerful_list)
            
            # Verify top candidates
            for idx in top_indices:
                if time.time() > deadline - 5:
                    break
                    
                N_cand = int(idx)
                
                count, exact_pairs = _count_pairs_for_N(N_cand, powerful_list, powerful_set)
                
                if count > best_count:
                    best_count = count
                    best_N = N_cand
                    best_pairs = exact_pairs
                    
                # Early termination: conv[idx] is count of ordered pairs
                # Unordered pairs <= ceil(conv[idx]/2)
                # If conv[idx] < 2 * best_count, it cannot beat best_count
                if conv[idx] < 2.0 * best_count:
                    break
            
            # Update best result
            if best_count > len(best_result['pairs']):
                best_result = {'N': best_N, 'pairs': best_pairs}
                
            # If we succeeded with a large M, it's likely the best.
            # Smaller M will have fewer powerful numbers, hence lower max count.
            # So we can return immediately if we tried a large M.
            # But just in case, let's return the best found so far.
            # Since we iterate M descending, the first successful large M is best.
            return best_result

        except MemoryError:
            continue
        except Exception as e:
            # Fallback to next smaller M
            continue

    return best_result

if __name__ == "__main__":
    res = construct()
    print(res['N'], len(res['pairs']))
