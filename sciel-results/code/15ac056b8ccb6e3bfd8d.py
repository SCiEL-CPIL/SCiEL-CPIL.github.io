import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Structural Change: Use FFT to estimate sum counts
    # Limit for powerful number generation
    L = 10**7
    
    # Generate powerful numbers up to L
    # Powerful numbers are of the form i^2 * j^3
    pows = set([1])
    max_i = math.isqrt(L)
    
    # Precompute cubes
    max_j = int(round(L ** (1/3))) + 2
    cubes = [j ** 3 for j in range(1, max_j + 1)]
    
    for i in range(1, max_i + 1):
        sq = i * i
        if sq > L:
            break
        max_j_local = int(round((L // sq) ** (1/3)))
        if max_j_local > max_j:
            max_j_local = max_j
        for j in range(1, max_j_local + 1):
            val = sq * cubes[j-1]
            if val <= L:
                pows.add(val)
            else:
                break
        if time.time() > deadline - 30:
             break

    P_list = sorted(pows)
    P_set = set(P_list)
    K = len(P_list)
    
    if K == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # FFT Autocorrelation
    # We need array size M >= 2 * max(P) to avoid wrap-around for sums < 2*max(P)
    # max(P) <= L. So M >= 2L.
    # Next power of 2 >= 2L
    M = 1
    while M < 2 * L:
        M *= 2
        
    # Create indicator array
    A = np.zeros(M, dtype=np.float64)
    for p in P_list:
        A[p] = 1.0
        
    # Compute FFT
    F = np.fft.fft(A)
    # Autocorrelation: IFFT(|F|^2)
    # The result at index k is sum A[i] * A[k-i]
    # We are interested in sums s = a + b.
    # In linear convolution/autocorrelation, the count for sum s is at index s.
    C = np.fft.ifft(F * np.conj(F)).real
    # Due to numerical precision, round to nearest integer
    C = np.round(C).astype(np.int64)
    
    if time.time() > deadline - 10:
        # Fallback to simple search if FFT took too long (unlikely)
        # Just pick a small N
        return {'N': 2, 'pairs': [[1, 1]]}

    # Find top candidates
    # We want max number of pairs a <= b.
    # C[s] counts ordered pairs (a,b) with a+b=s.
    # Number of distinct pairs a<=b is (C[s] + (1 if s/2 in P_set else 0)) // 2
    # We can just look at top indices of C.
    # Limit search to sums <= 2L (which is the valid range for our generated pows)
    # Actually, sums can go up to 2L.
    
    # Get top 100 candidate sums
    # We only care about s <= 2L
    valid_range = 2 * L
    top_indices = np.argpartition(C[:valid_range], -100)[-100:]
    
    best_N = 0
    best_pairs = []
    best_score = 0
    
    for s in top_indices:
        if s > 2 * L:
            continue
        # Verify and count pairs
        pairs = []
        for a in P_list:
            if a > s:
                break
            b = s - a
            if b < a:
                continue
            if b in P_set:
                pairs.append([int(a), int(b)])
        
        score = len(pairs)
        if score > best_score:
            best_score = score
            best_N = int(s)
            best_pairs = pairs
            
        if time.time() > deadline - 5:
            break
            
    if not best_pairs:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}

    return {'N': best_N, 'pairs': best_pairs}
