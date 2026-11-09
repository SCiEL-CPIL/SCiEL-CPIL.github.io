import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    N = 10000000

    # Compute sigma(n) for all n up to N using NumPy sieve
    sigma = np.zeros(N + 1, dtype=np.int64)

    for i in range(1, N + 1):
        if time.time() > deadline - 30.0:
            # If sieve is too slow, use fallback
            return [11, 6]
        sigma[i::i] += i

    if time.time() > deadline - 25.0:
        return [11, 6]

    # Group numbers by their sigma value
    # For each sigma value s, we have a list of n values with sigma(n) = s
    # We want to find pairs (a, b) in the same group where |a/b - 3/2| is minimized
    
    # Strategy: For each b, check if there's an a = round(3b/2) in the same sigma-fiber
    # More efficiently: iterate over b, compute target a = 3b//2 or (3b+1)//2, 
    # check if sigma(a) == sigma(b)
    
    # But this is O(N) which should be fast enough
    
    best_err_num = 1  # Start with error 1/1 (impossible)
    best_err_den = 1
    best_a = None
    best_b = None
    
    # For each b from 1 to N, check a values near 3b/2
    # a must satisfy |a/b - 3/2| < current best
    # i.e., |2a - 3b| < 2b * best_err_num / best_err_den
    
    # Start with a reasonable upper bound and refine
    # Check b in descending order (larger b gives smaller potential error)
    
    for b in range(N, 0, -1):
        if time.time() > deadline - 5.0:
            break
            
        sb = sigma[b]
        
        # Target a ≈ 3b/2
        # Check a values: floor(3b/2), ceil(3b/2), and nearby
        # |a/b - 3/2| = |2a - 3b| / (2b)
        # We want |2a - 3b| to be small
        
        # The minimum possible |2a - 3b| for integer a is 0 if 3b is even, 1 if 3b is odd
        # But we need a to be in [1, N] and sigma(a) == sigma(b)
        
        # Check a values around 3b/2
        # a_min = max(1, (3b - 2*best_err_num*b//best_err_den) // 2)  # rough
        # a_max = min(N, (3b + 2*best_err_num*b//best_err_den) // 2)  # rough
        
        # Simpler: just check a few values around 3b/2
        target_a = 3 * b // 2
        if 3 * b % 2 == 1:
            # 3b is odd, so 3b/2 is x.5
            candidates = [target_a, target_a + 1]
        else:
            # 3b is even
            candidates = [target_a]
        
        # Also check neighbors to be safe
        for offset in range(-5, 6):
            a = target_a + offset
            if 1 <= a <= N and a != b:  # a != b to avoid zero error
                if sigma[a] == sb:
                    # Compute error |a/b - 3/2| = |2a - 3b| / (2b)
                    k = abs(2 * a - 3 * b)
                    if k == 0:
                        continue  # Invalid: zero error
                    err_num = k
                    err_den = 2 * b
                    
                    # Compare with best
                    if err_num * best_err_den < best_err_num * err_den:
                        best_err_num = err_num
                        best_err_den = err_den
                        best_a = a
                        best_b = b
    
    if best_a is not None:
        return [int(best_a), int(best_b)]
        
    # Fallback
    return [11, 6]
