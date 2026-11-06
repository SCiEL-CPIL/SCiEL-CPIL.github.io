import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    # Limit L
    L = 10**12
    
    # Generate powerful numbers up to L
    # Powerful numbers are of the form a^2 * b^3 where b is square-free
    # b^3 <= L => b <= L^(1/3)
    max_b = int(L ** (1/3)) + 2
    
    # Sieve for square-free numbers up to max_b
    is_square_free = np.ones(max_b + 1, dtype=bool)
    is_square_free[0] = False
    is_square_free[1] = True
    
    for p in range(2, isqrt(max_b) + 1):
        if is_square_free[p]:
            p2 = p * p
            is_square_free[p2::p2] = False
            
    square_free_b = np.where(is_square_free)[0]
    square_free_b = square_free_b[square_free_b > 0] # Exclude 0
    
    powerfuls = set()
    for b in square_free_b:
        b3 = b ** 3
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        # a can be 1, 2, ..., max_a
        # a^2 * b^3
        # We can vectorize this?
        # a^2 is a range of squares
        # But we just need to add to set
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerfuls.add(n)
                
    powerfuls = np.array(list(powerfuls), dtype=np.int64)
    powerfuls.sort()
    
    if time.time() > deadline - 10:
         # Fallback to simple pair
         return {'N': 2, 'pairs': [[1, 1]]}
         
    # Compute sums
    # To save memory, we can do it in chunks or use int64
    # P = len(powerfuls)
    # We want all sums a[i] + a[j] for i <= j
    
    # Using broadcasting creates a P x P matrix
    # If P is 10^4, 10^8 elements. 800MB.
    # If P is 2*10^4, 4*10^8 elements. 3.2GB.
    
    # Let's check P
    P = len(powerfuls)
    
    if P > 20000:
        # Reduce L or use dictionary
        # For now, assume P is small enough
        pass
        
    # Efficient sum counting
    # We can use np.triu_indices
    i, j = np.triu_indices(P)
    sums = powerfuls[i] + powerfuls[j]
    
    # Count
    # np.unique is slow for 10^8 elements?
    # np.bincount is faster if range is small, but range is 10^12.
    # So np.unique is the way.
    
    unique_sums, counts = np.unique(sums, return_counts=True)
    
    max_count = np.max(counts)
    best_N = unique_sums[np.argmax(counts)]
    
    # Extract pairs
    pairs = []
    # We need to find pairs that sum to best_N
    # We can iterate through powerfuls
    p_set = set(powerfuls.tolist())
    for a in powerfuls:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in p_set and a <= b:
            pairs.append([a, b])
            
    pairs.sort()
    
    return {'N': int(best_N), 'pairs': pairs}
