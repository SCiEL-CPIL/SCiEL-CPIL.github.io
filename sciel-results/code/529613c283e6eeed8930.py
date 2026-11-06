import numpy as np
from math import isqrt

def construct():
    # Set L to maximize the number of powerful numbers while keeping memory and time in check
    # Memory for counts array: 2 * L * 4 bytes (int32)
    # For L = 10^8, memory is ~800 MB.
    # Time for M^2/2 ops: M ~ 1.47 * sqrt(10^8) ~ 14700. M^2/2 ~ 10^8 ops.
    # Python loop: ~10s.
    L = 100 * 10**6
    
    # Generate square-free numbers up to L^(1/3)
    max_b = int(L ** (1/3)) + 1
    is_sq_free = [True] * (max_b + 1)
    is_sq_free[0] = False
    is_sq_free[1] = True
    for i in range(2, isqrt(max_b) + 1):
        if is_sq_free[i]:
            sq = i * i
            for j in range(sq, max_b + 1, sq):
                is_sq_free[j] = False
    square_free = [i for i in range(1, max_b + 1) if is_sq_free[i]]
    
    powerful_set = set()
    for b in square_free:
        b3 = b ** 3
        if b3 > L: break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            powerful_set.add(a * a * b3)
            
    P = sorted(powerful_set)
    M = len(P)
    
    # Use numpy array for counts to save memory and speed up
    max_sum = 2 * L
    counts = np.zeros(max_sum + 1, dtype=np.int32)
    
    # Count sums
    # Python loop is slow for 10^8 iterations.
    # We can try to vectorize?
    # P is a list.
    # We can do:
    # for i in range(M):
    #    counts[P[i] + P[i:]] += 1  # This doesn't work directly with numpy on slices like this efficiently for accumulation
    # Actually, P[i] + P[i:] creates a new array.
    # np.add.at(counts, P[i] + P[i:], 1) is better.
    
    P_np = np.array(P, dtype=np.int64)
    
    for i in range(M):
        # P[i] + P[i:]
        sums = P[i] + P_np[i:]
        np.add.at(counts, sums, 1)
        
    # Find best N
    best_count = np.max(counts)
    best_N = np.argmax(counts)
    
    # Collect pairs
    pairs = []
    ps_set = powerful_set
    # Convert best_N to int
    best_N_int = int(best_N)
    for i in range(M):
        a = P[i]
        b = best_N_int - a
        if b < a: continue
        if b in ps_set:
            pairs.append([a, b])
            
    return {'N': best_N_int, 'pairs': pairs}
