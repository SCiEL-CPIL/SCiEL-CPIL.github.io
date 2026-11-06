import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # 1. Determine L_max based on available time/memory
    # We want to maximize L_max to find more pairs.
    # FFT size N_fft must be a power of 2 >= 2 * L_max.
    # Let's try L_max = 10^8. 2*L_max = 2*10^8.
    # Next power of 2: 2^28 = 268,435,456.
    # Memory for complex64: 268M * 8 bytes = 2.1 GB.
    # Memory for complex128: 268M * 16 bytes = 4.2 GB.
    # Let's use complex64 if possible, or stick to a smaller L_max if memory is a concern.
    # Given standard constraints, let's aim for L_max = 5 * 10^7 to be safe with complex128 or use complex64 for 10^8.
    
    L_max = 10**8
    
    # Generate powerful numbers up to 2 * L_max for verification set
    # and up to L_max for the FFT array.
    
    def gen_powerfuls(limit):
        pows = set()
        max_b = int(limit ** (1.0/3.0)) + 2
        for b in range(1, max_b + 1):
            b3 = b * b * b
            if b3 > limit:
                break
            max_a = isqrt(limit // b3)
            for a in range(1, max_a + 1):
                n = a * a * b3
                if n <= limit:
                    pows.add(n)
        return sorted(pows)
    
    powerfuls_list = gen_powerfuls(L_max)
    powerfuls_set = set(gen_powerfuls(2 * L_max)) # Set for verification up to 2*L_max
    
    if time.time() > deadline - 60:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # 2. FFT Convolution
    # Create indicator array
    N_fft = 1
    while N_fft < 2 * L_max + 1:
        N_fft *= 2
        
    A = np.zeros(N_fft, dtype=np.complex64)
    for p in powerfuls_list:
        if p < N_fft:
            A[p] = 1.0
            
    # FFT
    F = np.fft.fft(A)
    F2 = F * F
    C = np.fft.ifft(F2)
    C = np.real(C)
    
    # C[n] approximates count of pairs (a,b) with a+b=n, a,b <= L_max
    # We are interested in n in [2, 2*L_max]
    
    # Find top candidates
    # We want to maximize the count.
    # Slice C to relevant range
    C_relevant = C[2:2*L_max+1]
    
    # Get indices of top 100 values
    k = 100
    if len(C_relevant) < k:
        k = len(C_relevant)
        
    top_indices = np.argpartition(C_relevant, -k)[-k:]
    
    # Sort them by value descending
    top_indices = top_indices[np.argsort(C_relevant[top_indices])[::-1]]
    
    best_N = 2
    best_pairs = [[1, 1]]
    best_count = 1
    
    # 3. Verify top candidates
    # For each candidate N, count exact pairs using powerfuls_set
    # Note: The FFT only counted pairs where BOTH a and b are <= L_max.
    # If N is close to 2*L_max, it's possible that a > L_max or b > L_max.
    # However, since powerful numbers are sparse, the number of powerfuls in (L_max, 2*L_max] is small.
    # For N in the middle, both a and b are likely < L_max.
    # To be rigorous, we should check if N-a is in powerfuls_set (which goes up to 2*L_max).
    # But we must ensure we don't double count or miss pairs where a > L_max.
    # My loop `for a in powerfuls_list` only iterates a <= L_max.
    # If the optimal pair has a > L_max, we will miss it in the verification loop if we only iterate up to L_max.
    # But we generated powerfuls_list up to L_max.
    # If N is large, say 1.8 * L_max, and a = 1.9 * L_max (powerful), b = -0.1 * L_max (impossible).
    # If a > L_max, then b = N - a < N - L_max.
    # If N <= 2*L_max, b <= L_max.
    # So if we iterate a from powerfuls_list (<= L_max), we cover all pairs where a <= L_max.
    # Do we miss pairs where a > L_max?
    # If a > L_max, then b = N - a < N - L_max.
    # Since N <= 2*L_max, b < L_max.
    # So b is in powerfuls_list.
    # If we iterate a in powerfuls_list, we will find the pair (b, a) when a_loop = b.
    # Then we check if N - b = a is in powerfuls_set.
    # Since a <= N <= 2*L_max, a is in powerfuls_set.
    # So iterating a in powerfuls_list (sorted) and checking N-a in powerfuls_set covers ALL pairs {a,b} where a+b=N and a,b <= 2*L_max.
    # We just need to ensure we count each pair once.
    # By enforcing a <= b in the counting, we avoid double counting.
    
    for idx in top_indices:
        N = int(idx + 2) # Offset by 2 because C_relevant started at index 2
        
        if time.time() > deadline - 20:
            break
            
        pairs = []
        for a in powerfuls_list:
            if a > N // 2:
                break
            b = N - a
            if b in powerfuls_set:
                pairs.append([a, b])
                
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_N = N
            best_pairs = pairs
            
    if not best_pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    best_pairs.sort()
    return {'N': best_N, 'pairs': best_pairs}
