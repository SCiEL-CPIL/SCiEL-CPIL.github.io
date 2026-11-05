import time
import numpy as np

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 60.0  # seconds
    max_time_allowed = deadline - 5.0  # Safety margin

    N = 1000000
    LOWER = 10000
    UPPER = N
    
    # Step 1: Compute Smallest Prime Factor (SPF) for all numbers up to N
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Step 2: Compute powerful part for each number using optimized approach
    pow_part = [1] * (N + 2)
    
    # Process sequentially but with better optimization
    for i in range(2, N + 1):
        p = spf[i]
        m = i
        e = 0
        while m % p == 0:
            m //= p
            e += 1
        
        if e >= 2:
            pow_part[i] = pow_part[m] * (p ** e)
        else:
            pow_part[i] = pow_part[m]
    
    # Check time
    if time.time() - start_time > max_time_allowed:
        return 10000

    # Step 3: Find n in [10000, 1000000] that maximizes pow_part[n] * pow_part[n+1]
    # Use numpy for faster array operations
    pp_array = np.array(pow_part[LOWER:N+1], dtype=np.int64)
    scores = pp_array[:-1] * pp_array[1:]
    
    max_idx = np.argmax(scores)
    max_score = scores[max_idx]
    best_n = LOWER + max_idx
    
    return best_n
