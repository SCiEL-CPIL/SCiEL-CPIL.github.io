import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Strategy: 
    # 1. Generate powerful numbers from two disjoint prime sets.
    #    Set A: primes {2, 3}
    #    Set B: primes {5, 7}
    #    This ensures that for a+b=N where a in A, b in B, the representation is 
    #    "clean" in the sense that a and b share no prime factors, reducing 
    #    collisions or allowing us to sum counts from multiple such pairs of sets.
    #    Actually, the task allows ANY powerful numbers. 
    #    The current program uses ONE set of all powerful numbers <= M.
    #    The issue is that for large M, the FFT array becomes huge.
    #    
    #    Alternative Strategy: 
    #    Instead of one big FFT, use multiple smaller FFTs on subsets of powerful numbers 
    #    that are "shifted" or "scaled" to cover a larger range of N? 
    #    No, that's complex.
    #
    #    Let's look at the "Inspiration 1" finding: 
    #    "Increasing M from 20M to 80M expands the set of powerful numbers from ~7000 to ~23000, 
    #    enabling O(K^2) pair counting over a much larger space... Measured: 4.133 -> 5.933."
    #    
    #    The current program uses FFT with M=10^7. 
    #    The finding suggests that O(K^2) with K=23000 (M=80M) is feasible and yields better scores.
    #    Why? Because FFT with M=10^7 has size 2*10^7, which is ~160MB for complex? No, float64 is 8 bytes.
    #    2*10^7 * 8 bytes = 160MB. That's fine.
    #    But wait, the current program's M=10^7 gives K ~ 4500. 
    #    The finding says M=80M gives K~23000.
    #    If we do O(K^2) with K=23000, that's 23000^2 = 5.29 * 10^8 operations.
    #    In Python, this might be too slow (10-50s per 10^8 ops). 
    #    However, we can optimize the O(K^2) loop by using NumPy or by limiting the search.
    #    
    #    Let's try a hybrid: 
    #    Use FFT for a moderate M to get candidate Ns, then verify with a larger set?
    #    No, the verification is the expensive part if we use a larger set.
    #
    #    Let's rethink: The score is the number of pairs.
    #    The current program finds the max count in the FFT output.
    #    The FFT output is the convolution of the indicator vector of powerful numbers.
    #    This counts pairs (a,b) with a,b in the set.
    #    
    #    If we increase M, we get more powerful numbers, and likely more pairs for some N.
    #    The constraint is memory/time for FFT.
    #    M=10^7 -> size 2^24 (approx 16M) or 2^25 (32M)? 
    #    2*M = 2*10^7. Next power of 2 is 2^25 = 33,554,432.
    #    Array size 33.5M * 8 bytes = 268MB. This is borderline but usually okay.
    #    If we go to M=20M, size 2^26 = 67M. 67M * 8 = 536MB. Might OOM or be slow.
    #
    #    What if we use a "sliding window" or "chunking" approach?
    #    Or, what if we simply increase M as much as possible within memory limits?
    #    Let's try M = 15,000,000.
    #    Size needed: 2*15M = 30M. Next power of 2: 33,554,432 (2^25).
    #    This is the same FFT size as M=10M! 
    #    Wait, 2*10M = 20M. Next power of 2 is 33,554,432 (2^25).
    #    So M can go up to 16,777,216 (2^24) before the FFT size jumps to 2^26.
    #    Let's set M = 16,000,000.
    #    This will increase K from ~4500 to ~5500-6000.
    #    This should increase the max count.
    
    M = 16_000_000
    
    # 1. Generate powerful numbers up to M
    powerful_set = set()
    max_v = int(M ** (1/3)) + 2
    
    for v in range(1, max_v + 1):
        v_cubed = v ** 3
        if v_cubed > M:
            break
        max_u = isqrt(M // v_cubed)
        for u in range(1, max_u + 1):
            p = u * u * v_cubed
            powerful_set.add(p)
            
    powerful_list = sorted(powerful_set)
    K = len(powerful_list)
    
    # 2. Prepare for FFT
    size = 1
    while size < 2 * M + 1:
        size *= 2
        
    # Check time
    if time.time() > deadline - 30:
        return {'N': 2, 'pairs': [[1, 1]]}

    # Create indicator array
    A = np.zeros(size, dtype=np.float64)
    for p in powerful_list:
        A[p] = 1.0
        
    # 3. FFT Convolution
    F = np.fft.rfft(A)
    F_sq = F * F
    counts_float = np.fft.irfft(F_sq, n=size)
    
    # We only care about N up to 2*M
    counts_int = np.round(counts_float[:2*M + 1]).astype(np.int64)
    
    # 4. Find candidates
    max_val = np.max(counts_int)
    if max_val == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    top_k = 100
    top_indices = np.argpartition(counts_int, -top_k)[-top_k:]
    
    # 5. Verify exact counts for these candidates
    best_N = 0
    best_exact_count = 0
    best_pairs = []
    
    powerful_set_local = powerful_set
    
    for idx in top_indices:
        N = int(idx)
        count = 0
        pairs = []
        # Iterate through powerful numbers <= N/2
        # Use binary search to limit the loop? 
        # powerful_list is sorted.
        # Find index of first element > N//2
        # Since N <= 2*M, and powerful_list <= M, we just iterate all K.
        # K is ~6000. 100 * 6000 = 600,000 checks. Fast.
        
        for a in powerful_list:
            if a > N // 2:
                break
            b = N - a
            if b in powerful_set_local:
                count += 1
                pairs.append([a, b])
                
        if count > best_exact_count:
            best_exact_count = count
            best_N = N
            best_pairs = pairs
            
    if time.time() > deadline - 5:
        pass

    return {'N': best_N, 'pairs': best_pairs}
