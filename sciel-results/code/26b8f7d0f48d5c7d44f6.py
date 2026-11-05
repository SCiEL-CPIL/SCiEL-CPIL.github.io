import numpy as np
import scipy.signal
from math import isqrt

def construct():
    # Set the limit for generating powerful numbers
    # L = 10^7 allows checking sums up to 2*10^7
    # This is a balance between computation time/memory and finding a high-scoring N
    L = 10**7
    
    # 1. Generate all powerful numbers up to L
    powerful_set = set()
    # b^3 <= L  =>  b <= L^(1/3)
    max_b = int(L ** (1/3)) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        # a^2 * b^3 <= L  =>  a^2 <= L / b^3  =>  a <= sqrt(L / b^3)
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerful_set.add(n)
    
    # Convert to sorted list for pair reconstruction
    powerful_list = sorted(powerful_set)
    
    # 2. Prepare array for FFT convolution
    # We want to count representations N = a + b
    # arr[i] = 1 if i is powerful, 0 otherwise
    arr = np.zeros(L, dtype=np.float32)
    for p in powerful_list:
        arr[p] = 1.0
        
    # 3. Compute convolution
    # The convolution of arr with itself gives the number of ordered pairs (a, b) such that a + b = k
    # We use fftconvolve for speed
    # mode='full' gives the full linear convolution
    try:
        conv = scipy.signal.fftconvolve(arr, arr, mode='full')
    except Exception:
        # Fallback to smaller L if memory error or other issue
        L = 5 * 10**6
        arr = np.zeros(L, dtype=np.float32)
        for p in powerful_list:
            if p < L:
                arr[p] = 1.0
        conv = scipy.signal.fftconvolve(arr, arr, mode='full')
        
    # 4. Find the N that maximizes the number of distinct pairs
    # conv[k] is the count of ordered pairs (a, b) with a + b = k
    # The number of distinct pairs a <= b is (conv[k] + D(k)) / 2
    # where D(k) = 1 if k is even and k/2 is powerful, else 0
    # Since D(k) is at most 1, the index of the maximum of conv is a very strong candidate
    # We check the global maximum and a few neighbors to be safe
    
    # Find the index of the maximum value in conv
    # conv is an array of size 2*L - 1
    # Index k corresponds to sum k
    # We require N >= 2
    
    # Get the top 5 indices to handle ties or slight variations due to D(k)
    # np.argsort is O(N log N), which might be slow for 20M elements in Python loop context?
    # Actually, np.argpartition is O(N).
    
    # Let's just take the argmax. The probability that a neighbor with D=1 beats the global max with D=0
    # is non-zero if conv[global_max] == conv[neighbor].
    
    # To be robust, let's find the max value and check indices around it.
    # Or simply, since D(k) is small, the global max of conv is almost certainly the best.
    
    idx_max = np.argmax(conv)
    
    # Check a small window around idx_max to account for the D(k) term
    # and to ensure we pick the best N
    start_idx = max(2, idx_max - 10)
    end_idx = min(len(conv), idx_max + 10)
    
    best_N = -1
    best_score = -1
    
    for N in range(start_idx, end_idx):
        # Calculate distinct pairs count
        ordered_count = conv[N]
        # Check if N/2 is powerful
        if N % 2 == 0:
            half = N // 2
            if half in powerful_set:
                distinct_count = (ordered_count + 1) / 2
            else:
                distinct_count = ordered_count / 2
        else:
            distinct_count = ordered_count / 2
            
        if distinct_count > best_score:
            best_score = distinct_count
            best_N = N
            
    if best_N == -1:
        best_N = 2
        
    # 5. Reconstruct the pairs for best_N
    pairs = []
    half = best_N // 2
    for a in powerful_list:
        if a > half:
            break
        b = best_N - a
        if b in powerful_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
