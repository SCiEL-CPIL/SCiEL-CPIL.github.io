import numpy as np
from collections import defaultdict
import bisect
import time

def construct():
    N = 10_000_000
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe
    
    # 1. Compute sigma using NumPy
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        if time.time() > deadline:
            break
            
    # 2. Group numbers by sigma
    groups = defaultdict(list)
    for n in range(1, N + 1):
        groups[sigma[n]].append(n)
        
    if time.time() > deadline:
        return [1, 1] # Fallback, though likely invalid or poor
        
    # 3. Search for best pair
    best_error = float('inf')
    best_a = 1
    best_b = 1
    
    # Filter groups with size > 1
    # And maybe size > 2 to ensure a != b is possible?
    # If size == 1, a=b, error=0.5.
    # We want error < 0.5.
    
    for s, L in groups.items():
        if len(L) < 2:
            continue
            
        # L is sorted
        # We want a/b close to 1.5
        # a, b in L
        
        # Optimization: if the range of ratios doesn't cover 1.5, skip?
        # min_ratio = L[0] / L[-1]
        # max_ratio = L[-1] / L[0]
        # If 1.5 not in [min_ratio, max_ratio], skip.
        # But we can just do the search.
        
        for b in L:
            target = 1.5 * b
            pos = bisect.bisect_left(L, target)
            
            # Check pos
            if pos < len(L):
                a = L[pos]
                if a != b:
                    err = abs(a / b - 1.5)
                    if err < best_error:
                        best_error = err
                        best_a = a
                        best_b = b
                        if best_error == 0:
                            return [best_a, best_b]
            
            # Check pos-1
            if pos > 0:
                a = L[pos - 1]
                if a != b:
                    err = abs(a / b - 1.5)
                    if err < best_error:
                        best_error = err
                        best_a = a
                        best_b = b
                        if best_error == 0:
                            return [best_a, best_b]
                        
        if time.time() > deadline:
            break
            
    return [int(best_a), int(best_b)]
