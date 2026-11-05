import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for safety
    
    target = 123457
    
    # Precompute largest prime factor for all numbers up to target
    # Using a NumPy-based sieve approach for speed
    lpf = np.ones(target + 1, dtype=np.int64)
    for i in range(2, target + 1):
        if lpf[i] == 1:  # i is prime
            lpf[i::i] = i
    
    # Convert to list for faster indexing in the Python scan loop
    lpf_list = lpf.tolist()
    
    # Now find a, b with a + b = target minimizing max(lpf[a], lpf[b])
    best_score = float('inf')
    best_a, best_b = 1, target - 1
    
    for a in range(1, target // 2 + 1):
        b = target - a
        score = max(lpf_list[a], lpf_list[b])
        if score < best_score:
            best_score = score
            best_a, best_b = a, b
            if best_score == 2:
                break
    
    if time.time() > deadline:
        pass
    
    return [best_a, best_b]
