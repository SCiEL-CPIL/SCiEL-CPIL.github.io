import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    N = 10_000_000
    
    # Sieve for sigma
    sigma = np.ones(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        if time.time() > deadline - 10:
            break
        sigma[d::d] += d
    
    # Group indices by sigma value
    # We need to find pairs (a,b) with sigma[a]==sigma[b] and a/b close to 3/2
    # For efficiency, for each b, we want a ≈ 3b/2
    # So for each b, check a = round(3b/2) and nearby values
    
    # Better: iterate over b, compute target a = 3b/2, check if sigma[round(3b/2)] == sigma[b]
    # But we need exact minimization of |a/b - 3/2|
    
    # Strategy: for each b from 2 to N, the ideal a is 3b/2.
    # Check a values near 3b/2 that have the same sigma as b.
    
    # Even better: build a dict mapping sigma_value -> list of n
    # But that could use too much memory for 10^7 entries
    
    # Alternative: for each b, compute a_candidate = round(3*b/2), check if in range and sigma match
    # Then also check a_candidate +/- small range
    
    best_err_num = None
    best_err_den = None
    best_a = None
    best_b = None
    
    # For each b, the closest a to 3b/2 with same sigma
    # a must be integer, so 3b/2 may not be integer
    # If b is even: a_ideal = 3b/2 (integer)
    # If b is odd: a_ideal = 3b/2 (half-integer), so check floor and ceil
    
    for b in range(2, N + 1):
        if time.time() > deadline - 5:
            break
        
        sb = sigma[b]
        
        # Ideal a = 3b/2
        if b % 2 == 0:
            a_ideal = 3 * b // 2
            candidates = [a_ideal]
        else:
            a_ideal_low = (3 * b) // 2
            a_ideal_high = a_ideal_low + 1
            candidates = [a_ideal_low, a_ideal_high]
        
        for a in candidates:
            if a < 1 or a > N:
                continue
            if a == b:
                continue
            if sigma[a] == sb:
                # Compute |a/b - 3/2| = |2a - 3b| / (2b)
                err_num = abs(2 * a - 3 * b)
                err_den = 2 * b
                if err_num == 0:
                    continue  # invalid
                if best_err_num is None:
                    best_err_num = err_num
                    best_err_den = err_den
                    best_a = a
                    best_b = b
                else:
                    # Compare err_num/err_den vs best_err_num/best_err_den
                    # err_num * best_err_den vs best_err_num * err_den
                    lhs = err_num * best_err_den
                    rhs = best_err_num * err_den
                    if lhs < rhs:
                        best_err_num = err_num
                        best_err_den = err_den
                        best_a = a
                        best_b = b
    
    if best_a is None:
        # Fallback: find any pair with same sigma
        # Use the original [11, 6]
        return [11, 6]
    
    return [int(best_a), int(best_b)]
