import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Set N_MAX
    # Based on analysis, 5e7 should be fast with NumPy
    N_MAX = 50000000 
    
    # 1. Generate powerful numbers up to N_MAX
    powerful_set = set()
    # v^3 <= N_MAX => v <= N_MAX^(1/3)
    max_v = int(N_MAX ** (1/3)) + 1
    
    # We need v to be square-free? No, powerful numbers are of the form u^2 v^3 where v is square-free?
    # Actually, any powerful number can be written as $a^2 b^3$ where $b$ is square-free.
    # Or simply generate all $x^2 y^3 \le N_MAX$.
    # To avoid duplicates, we can just use a set.
    
    # Optimization: Iterate y (cube part) and x (square part)
    # y^3 <= N_MAX
    # x^2 <= N_MAX / y^3
    
    for y in range(1, max_v + 1):
        y3 = y ** 3
        if y3 > N_MAX:
            break
        max_x = isqrt(N_MAX // y3)
        for x in range(1, max_x + 1):
            val = x * x * y3
            powerful_set.add(val)
            
    P = sorted(powerful_set)
    
    # 2. Initialize counts array
    counts = np.zeros(N_MAX + 1, dtype=np.int32)
    
    # 3. Iterate and count
    # P is sorted
    # For each a in P, find b in P such that b >= a and a+b <= N_MAX
    # We can use bisect to find the range of b
    
    import bisect
    
    for a in P:
        if time.time() > deadline:
            break
        # b must be >= a and <= N_MAX - a
        # Find start index for b >= a
        start_idx = bisect.bisect_left(P, a)
        # Find end index for b <= N_MAX - a
        end_idx = bisect.bisect_right(P, N_MAX - a)
        
        if start_idx < end_idx:
            b_vals = P[start_idx:end_idx]
            # Vectorized addition
            counts[a + b_vals] += 1
            
    # 4. Find max count
    max_count = np.max(counts)
    if max_count == 0:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
        
    N_best = int(np.argmax(counts))
    
    # 5. Find pairs for N_best
    pairs = []
    # We need to find all a, b in P such that a+b = N_best and a <= b
    # We can iterate a in P up to N_best//2
    for a in P:
        if a > N_best // 2:
            break
        b = N_best - a
        if b in powerful_set and b >= a:
            pairs.append([a, b])
            
    return {'N': N_best, 'pairs': pairs}
