import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    
    # Range for S calculation needs to cover N_MAX + WINDOW
    MAX_VAL = N_MAX + WINDOW + 10
    
    # Primes < 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We want S[x] = product of prime factors of x that are < 24.
    # Let L[x] = product of prime factors of x that are >= 24.
    # Then S[x] = x // L[x].
    # We can compute L[x] by starting with L[x] = x and dividing out all small primes.
    
    L = np.arange(1, MAX_VAL + 1, dtype=np.int64)
    
    # For each small prime p, divide L[m] by p as many times as possible
    for p in primes:
        # Find all indices m where L[m] is divisible by p
        # Note: L[m] changes, so we must check divisibility dynamically or use a mask.
        # Efficient way: iterate through powers of p? No, just check L[m] % p == 0.
        # Since we divide L[m], subsequent checks for the same p must be on the updated L[m].
        
        # We can use a while loop for each prime p, processing multiples.
        # However, NumPy vectorization is key.
        # Let's do it in passes.
        # For a fixed p, we want to divide L[m] by p for all m where L[m] % p == 0.
        # We can create a mask. But the mask changes as we divide.
        # A simpler vectorized approach for "divide out all factors of p":
        # While there exists m such that L[m] % p == 0:
        #   mask = (L % p == 0)
        #   L[mask] //= p
        # This is efficient because the number of divisions is small (log_p(MAX_VAL)).
        
        while True:
            mask = (L % p == 0)
            if not np.any(mask):
                break
            L[mask] //= p
            
    # Now L[x] contains the part of x composed of primes >= 24.
    # The part composed of primes < 24 is S[x] = x // L[x].
    # We only need S[x] for x in [N_MIN+1, N_MAX+WINDOW].
    # But computing S for all x is fast now.
    X = np.arange(1, MAX_VAL + 1, dtype=np.int64)
    S = X // L
    
    # We need to find n in [N_MIN, N_MAX] minimizing distinct count in S[n+1..n+WINDOW]
    
    best_n = N_MIN
    best_count = float('inf')
    
    # Sliding window
    # Initial window for n = N_MIN: indices N_MIN+1 to N_MIN+WINDOW
    # S is 1-indexed in our array construction (index 0 is unused/1)
    # S[i] corresponds to value i.
    
    window_start = N_MIN + 1
    window_end = N_MIN + WINDOW + 1 # exclusive
    
    # Extract the window
    window_vals = S[window_start:window_end]
    
    # Count distinct values
    # Using a dictionary or np.unique?
    # np.unique is O(K log K) or O(K) with sort. Dictionary is O(K).
    # K=24 is small. Dictionary is fine.
    
    counts = {}
    for v in window_vals:
        counts[v] = counts.get(v, 0) + 1
        
    current_distinct = len(counts)
    
    if current_distinct < best_count:
        best_count = current_distinct
        best_n = N_MIN
        
    if best_count == 1:
        return best_n
        
    # Slide
    # Next n is N_MIN + 1
    # Remove S[N_MIN+1], Add S[N_MIN+WINDOW+1]
    
    for n in range(N_MIN + 1, N_MAX + 1):
        # The element leaving the window is at index n (which was n-1+1)
        val_out = S[n]
        counts[val_out] -= 1
        if counts[val_out] == 0:
            del counts[val_out]
            
        # The element entering the window is at index n + WINDOW
        val_in = S[n + WINDOW]
        counts[val_in] = counts.get(val_in, 0) + 1
        
        current_distinct = len(counts)
        
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
            if best_count == 1:
                return best_n
                
        # Time check
        if time.time() > deadline:
            break
            
    return best_n
