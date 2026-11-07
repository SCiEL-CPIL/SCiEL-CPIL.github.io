import numpy as np
import time

def construct():
    start_time = time.time()
    # Deadline: 85s to be safe against 90s kill
    deadline = start_time + 85.0
    
    N = 10000000
    
    # 1. Compute sigma(n) for all n <= N
    # Using the sieve method: sigma[j] += i for all j multiple of i
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # We iterate i from 1 to N
    # To avoid timeout, we check time periodically
    # However, checking time 10 million times is slow.
    # We can check every 100,000 iterations.
    
    for i in range(1, N + 1):
        if i % 100000 == 0:
            if time.time() > deadline - 10.0:
                # If we are running out of time, we cannot complete the sieve correctly.
                # We might return a suboptimal but valid result from what we have?
                # No, if sigma is incomplete, the pairs are invalid.
                # So we must finish or crash.
                # Given 90s limit and ~30s estimate, we should be fine.
                pass
        sigma[i::i] += i
        
    # 2. Build dictionary mapping sigma value to list of n
    d = {}
    # Iterate n from 1 to N
    # This loop is ~5-10s
    for n in range(1, N + 1):
        if time.time() > deadline - 5.0:
            break
        s = sigma[n]
        if s in d:
            d[s].append(n)
        else:
            d[s] = [n]
            
    # 3. Search for best pair
    best_err = float('inf')
    best_a, best_b = 1, 1
    
    # Iterate over all sigma values
    for L in d.values():
        if len(L) < 2:
            continue
            
        # L is sorted because we appended n in increasing order
        len_L = len(L)
        j = 0
        
        for i in range(len_L):
            b = L[i]
            # We want a such that a/b ~ 1.5 => 2a ~ 3b => a ~ 1.5b
            # Target a is approx 1.5 * b
            target = 1.5 * b
            
            # Advance j to the first element >= target
            # Since b increases, target increases, so j only moves forward
            while j < len_L and L[j] < target:
                j += 1
                
            # Check candidate at j
            if j < len_L:
                a = L[j]
                if a > b:
                    # Calculate error exactly
                    # err = |a/b - 1.5| = |2a - 3b| / (2b)
                    # We want to minimize this.
                    # Since b is fixed for this step, minimizing |2a - 3b| is sufficient
                    # but we need the global min over all b.
                    # So we compute the float error or a comparable metric.
                    # Using float is fine for comparison if we are careful, 
                    # or we can store (numerator, denominator) and compare fractions.
                    # For simplicity and speed, use float.
                    err = abs(a / b - 1.5)
                    if err < best_err:
                        best_err = err
                        best_a, best_b = a, b
                        
            # Check candidate at j-1
            if j > 0:
                a = L[j-1]
                if a > b:
                    err = abs(a / b - 1.5)
                    if err < best_err:
                        best_err = err
                        best_a, best_b = a, b
                        
    return [int(best_a), int(best_b)]
