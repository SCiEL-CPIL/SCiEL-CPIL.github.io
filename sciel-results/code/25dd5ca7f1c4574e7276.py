import time
import numpy as np

def construct():
    start_time = time.time()
    LIMIT = 90  # seconds (evaluator kills at 90s)
    MARGIN = 5  # seconds safety margin
    
    N_MAX = 1000001  # We need P up to 1000001 to evaluate n=1000000
    START = 10000
    
    # 1. Sieve for Smallest Prime Factor (SPF)
    # Using a list of ints
    spf = list(range(N_MAX + 1))
    for i in range(2, int(N_MAX**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N_MAX + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute Powerful Part P[k]
    # P[i] is the product of p^e for all prime powers p^e || i where e >= 2
    P = [1] * (N_MAX + 1)
    P[1] = 1
    
    for i in range(2, N_MAX + 1):
        p = spf[i]
        x = i // p
        if x % p == 0:
            y = x // p
            if y % p == 0:
                # i has p^3 or higher
                P[i] = P[x] * p
            else:
                # i has exactly p^2
                P[i] = P[x] * p * p
        else:
            # i has exactly p^1
            P[i] = P[x]
            
    # Check time
    if time.time() - start_time > LIMIT - MARGIN:
        return 10000

    # 3. Find n that maximizes P[n] * P[n+1]
    try:
        # Convert P to numpy array for fast vectorized multiplication
        P_arr = np.array(P, dtype=np.int64)
        
        # n ranges from START to 1000000 inclusive.
        # P[n] corresponds to indices START to 1000000
        # P[n+1] corresponds to indices START+1 to 1000001
        left = P_arr[START:1000001]
        right = P_arr[START+1:1000002]
        
        products = left * right
        
        max_idx = np.argmax(products)
        best_n = int(START + max_idx)
        
    except Exception:
        # Fallback to pure Python if numpy fails or overflows
        best_n = START
        best_score = -1
        for n in range(START, 1000001):
            score = P[n] * P[n + 1]
            if score > best_score:
                best_score = score
                best_n = n

    return best_n
