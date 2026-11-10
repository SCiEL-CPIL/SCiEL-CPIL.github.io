import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    N = 10**7
    
    # 1. Compute sigma(n) for n=1..N using vectorized NumPy sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve: for each i, add i to all multiples of i
    for i in range(1, N + 1):
        sigma[i::i] += i
    
    if time.time() > deadline - 5:
        return [11, 6]
    
    # 2. Search for best pair by iterating b and searching near 1.5*b
    best_a, best_b = 11, 6
    best_err = abs(11 / 6 - 1.5)  # ~0.333...
    
    # Iterate b from high to low - larger b tends to allow smaller relative errors
    for b in range(N, 0, -1):
        if time.time() > deadline - 2:
            break
        
        # Calculate the window of 'a' values to check
        # We want |a/b - 1.5| < best_err
        # => |a - 1.5*b| < 1.5 * best_err * b
        # But we also need to handle the case where best_err is still large (seed value)
        # In that case, cap the window to avoid O(N^2) behavior
        
        if best_err > 0.1:
            # Still using seed or similar - use a fixed reasonable window
            window = 2000
        else:
            # Use adaptive window based on current best error
            window = int(1.5 * b * best_err) + 1
            # Cap window to avoid huge loops
            if window > 5000:
                window = 5000
            # Minimum window to ensure we check at least a few values
            if window < 10:
                window = 10
        
        a_center = int(1.5 * b)
        a_start = max(1, a_center - window)
        a_end = min(N, a_center + window)
        
        if a_end < a_start:
            continue
        
        # Vectorized check: find all a in [a_start, a_end] where sigma[a] == sigma[b]
        target_sigma = sigma[b]
        candidates = np.where(sigma[a_start:a_end + 1] == target_sigma)[0]
        
        for idx in candidates:
            a = a_start + idx
            # Check invalid condition: a/b == 1.5 exactly => 2a == 3b
            if 2 * a == 3 * b:
                continue
            
            # Compute error
            err = abs(a / b - 1.5)
            if err < best_err:
                best_err = err
                best_a = int(a)
                best_b = int(b)
    
    return [best_a, best_b]
