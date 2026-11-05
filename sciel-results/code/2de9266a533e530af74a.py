import time
from math import isqrt
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def time_left():
        return deadline - time.time()
        
    # Choose M
    # M = 30,000,000 allows FFT size 2^26 (67M) which is ~1GB memory
    M = 30_000_000
    
    # Generate powerful numbers up to M
    # n = a^2 * b^3, b square-free
    max_b = int(M ** (1.0 / 3.0)) + 2
    
    def is_square_free(n):
        if n < 2:
            return n == 1
        temp = n
        p = 2
        while p * p <= temp:
            if temp % (p * p) == 0:
                return False
            while temp % p == 0:
                temp //= p
            p += 1
        return True

    powerful = []
    
    for b in range(1, max_b + 1):
        if not is_square_free(b):
            continue
        b_cubed = b ** 3
        if b_cubed > M:
            break
        max_a = isqrt(M // b_cubed)
        for a in range(1, max_a + 1):
            powerful.append(a * a * b_cubed)
            
    powerful.sort()
    P = np.array(powerful, dtype=np.int64)
    P_set = set(powerful) # For pair reconstruction
    
    # Determine FFT size
    # We need to compute sums up to 2*M
    # Next power of 2 >= 2*M
    target_size = 2 * M
    fft_size = 1
    while fft_size < target_size:
        fft_size <<= 1
        
    # Create indicator array
    # A[i] = 1 if i is powerful, else 0
    # Note: powerful numbers are 1-indexed in value, but array is 0-indexed.
    # A[0] should be 0 (0 is not powerful in this context? 1 is powerful. 0 is not.)
    # The problem says "1 counts as powerful". 0 is not a sum of two positive powerful numbers usually?
    # "distinct pairs a <= b, a + b = N". Usually positive integers.
    # If a=0, is 0 powerful? "every prime factor appears squared". 0 has infinite prime factors?
    # Standard definition: Powerful numbers are positive integers.
    # So A[0] = 0.
    
    A = np.zeros(fft_size, dtype=np.float64)
    # P contains values 1..M
    A[P] = 1.0
    
    if time.time() > deadline - 30:
        # Fallback to smaller M if FFT setup is slow (unlikely)
        # Just return a small valid result
        return {'N': 2, 'pairs': [[1, 1]]}

    # FFT
    F = np.fft.fft(A)
    G = F * F
    C = np.fft.ifft(G)
    
    # C[k] is the number of ordered pairs (a,b) such that a+b=k
    # We need to round to nearest integer
    Counts = np.rint(C.real).astype(np.int64)
    
    # We want to maximize D(N) = (Counts[N] + n_eq(N)) / 2
    # n_eq(N) is 1 if N is even and N/2 is in P_set, else 0.
    
    # We only care about N up to 2*M
    # Find indices with high Counts
    # We can just scan the top few counts
    
    # To be efficient, we can find the max Counts value, then check neighbors?
    # Or just check all N from 2 to 2*M?
    # 2*M is 60,000,000. Iterating 60M times in Python is too slow (~30s).
    # We need to find the max efficiently.
    
    # Let's find the top 100 indices by Counts
    # argsort is O(N log N). N=67M. This might be slow.
    # Alternative: np.max and then search?
    # But the max Counts might not correspond to max D if n_eq is different.
    # However, n_eq is 0 or 1. Counts is usually >> 1.
    # So max Counts is a good proxy.
    
    # Let's just take the top 1000 indices from the FFT result
    # Using argpartition is O(N)
    top_k = 1000
    if len(Counts) > top_k:
        # We only care about indices 2..2*M
        # Indices 0, 1 are 0.
        # Let's restrict to 2..2*M
        valid_counts = Counts[2:2*M+1]
        # Get top k indices in valid_counts
        # Note: argpartition returns indices relative to valid_counts
        part_idx = np.argpartition(valid_counts, -top_k)[-top_k:]
        # Get actual values
        top_vals = valid_counts[part_idx]
        # Sort them to get the very best ones
        sorted_idx = np.argsort(top_vals)[::-1]
        top_indices = part_idx[sorted_idx] + 2 # Offset by 2
        
        # Check these candidates
        best_score = 0
        best_N = -1
        for idx in top_indices:
            N = int(idx)
            c = int(Counts[N])
            n_eq = 0
            if N % 2 == 0:
                half = N // 2
                if half in P_set:
                    n_eq = 1
            score = (c + n_eq) // 2
            if score > best_score:
                best_score = score
                best_N = N
    else:
        # Small array, just scan
        best_score = 0
        best_N = -1
        for N in range(2, 2*M+1):
            c = int(Counts[N])
            n_eq = 0
            if N % 2 == 0:
                half = N // 2
                if half in P_set:
                    n_eq = 1
            score = (c + n_eq) // 2
            if score > best_score:
                best_score = score
                best_N = N

    if best_N == -1:
        return {'N': 2, 'pairs': [[1, 1]]}

    # Reconstruct pairs for best_N
    pairs = []
    # Iterate through powerful numbers a <= best_N/2
    # P is sorted
    # We need a in P and best_N - a in P
    # And a <= best_N - a => a <= best_N/2
    
    limit = best_N // 2
    # Binary search or simple loop?
    # P is numpy array.
    # We can use searchsorted to find end of a <= limit
    end_idx = np.searchsorted(P, limit)
    
    for i in range(end_idx):
        a = int(P[i])
        b = best_N - a
        # Since a <= limit, a <= b
        if b in P_set:
            pairs.append([a, b])
            
    return {'N': int(best_N), 'pairs': pairs}
