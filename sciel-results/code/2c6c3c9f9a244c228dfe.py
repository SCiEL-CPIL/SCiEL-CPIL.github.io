import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Limit L
    # P ~ 2.17 * sqrt(L)
    # For L=1e9, P ~ 68000. P^2 ~ 4.6e9 ops.
    # Chunked approach.
    L = 10**9
    
    # 1. Generate powerful numbers up to L
    # n = u^2 * v^3 where v is square-free
    max_v = int(L ** (1/3)) + 2
    
    # Sieve for square-free numbers up to max_v
    is_sq_free = np.ones(max_v + 1, dtype=bool)
    is_sq_free[0] = False
    is_sq_free[1] = True
    for p in range(2, isqrt(max_v) + 1):
        if is_sq_free[p]:
            p2 = p * p
            is_sq_free[p2::p2] = False
            
    square_free_v = np.nonzero(is_sq_free)[0]
    # Filter v such that v^3 <= L
    square_free_v = square_free_v[square_free_v**3 <= L]
    
    powerfuls = set()
    powerfuls.add(1)
    
    for v in square_free_v:
        v3 = v**3
        max_u = isqrt(L // v3)
        if max_u < 1:
            continue
        # u from 1 to max_u
        # n = u^2 * v3
        # Vectorize u
        u_vals = np.arange(1, max_u + 1)
        n_vals = (u_vals**2) * v3
        # Add to set
        for n in n_vals:
            powerfuls.add(int(n))
            
    powerfuls_list = np.array(sorted(powerfuls), dtype=np.int64)
    P = len(powerfuls_list)
    
    # 2. Count sums using chunked vectorization
    # We want to count pairs (a, b) with a <= b such that a+b=N
    # We can count all pairs (a, b) and then adjust, or just count a <= b directly.
    # Direct a <= b is harder to vectorize simply.
    # Let's count all pairs (a, b) and store counts.
    # Then for a specific N, the number of pairs a<=b is:
    # If N is odd: count(N) / 2
    # If N is even: (count(N) - 1) / 2 + 1 if N/2 is powerful else (count(N))/2
    # Actually, simpler:
    # Just find the N with max count(N).
    # Then generate the pairs for that N by checking a <= N/2.
    
    # Chunking
    chunk_size = 2000  # Adjust based on memory
    counts = {}
    
    for i in range(0, P, chunk_size):
        end = min(i + chunk_size, P)
        a_chunk = powerfuls_list[i:end]
        
        # Calculate sums a_chunk + powerfuls_list
        # This creates a (chunk_size x P) array
        sums = a_chunk[:, None] + powerfuls_list[None, :]
        
        # Count frequencies
        # np.bincount is fast but requires non-negative integers
        # Max sum is 2*L = 2e9.
        # bincount on 2e9 length array is too big (2e9 * 8 bytes = 16GB).
        # So we cannot use bincount on the full range.
        # We need to use a dictionary or unique+count.
        # unique+count on 4e7 elements (2000*20000) might be slow.
        # Let's try smaller chunks or different approach.
        
        # Alternative: Use a dictionary for sums.
        # But adding 4e7 keys to a dict is slow.
        
        # Let's go back to FFT if P is small enough?
        # For L=1e8, P~20000. FFT size 2e8. Memory ~1.6GB (complex).
        # Maybe FFT is safer for L=1e8.
        # Let's try L=1e8 with FFT.
        
        pass # Abort chunked dict approach for now, switch to FFT logic below if L is small.
        
    # Let's restart strategy: Use FFT for L=1e8 or 2e8.
    # If L=2e8, P~30000.
    # FFT size 4e8. Memory ~3.2GB. Might be risky.
    # Let's try L=1e8 first.
    
    L = 10**8
    max_v = int(L ** (1/3)) + 2
    is_sq_free = np.ones(max_v + 1, dtype=bool)
    is_sq_free[0] = False
    is_sq_free[1] = True
    for p in range(2, isqrt(max_v) + 1):
        if is_sq_free[p]:
            p2 = p * p
            is_sq_free[p2::p2] = False
    square_free_v = np.nonzero(is_sq_free)[0]
    square_free_v = square_free_v[square_free_v**3 <= L]
    
    powerfuls = set([1])
    for v in square_free_v:
        v3 = v**3
        max_u = isqrt(L // v3)
        if max_u < 1: continue
        u_vals = np.arange(1, max_u + 1)
        n_vals = (u_vals**2) * v3
        for n in n_vals:
            powerfuls.add(int(n))
            
    powerfuls_list = np.array(sorted(powerfuls), dtype=np.int64)
    
    # Create indicator array
    A = np.zeros(2*L + 1, dtype=np.int32)
    A[powerfuls_list] = 1
    
    # Convolution
    # scipy.signal.fftconvolve
    from scipy.signal import fftconvolve
    C = fftconvolve(A, A, mode='full')
    
    # Find max
    # C[k] is number of ordered pairs (a,b) with a+b=k
    # We want max number of unordered pairs.
    # This is roughly C[k]/2.
    # Just find index of max C[k]
    best_N = int(np.argmax(C))
    
    # Extract pairs
    pairs = []
    p_set = set(powerfuls_list)
    for a in powerfuls_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in p_set and a <= b:
            pairs.append([int(a), int(b)])
            
    return {'N': best_N, 'pairs': pairs}
