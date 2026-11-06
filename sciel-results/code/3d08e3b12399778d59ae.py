import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # Sieve to find largest prime factor for all numbers up to N_MAX + 1
    # We need lpf[n+1] for n up to N_MAX, so we need up to N_MAX + 1
    limit = N_MAX + 1
    lpf = [0] * (limit + 1)
    
    # Standard sieve for largest prime factor
    # Initialize lpf for all numbers
    # For each prime p, mark its multiples with p
    # Since we iterate p in increasing order, the last mark is the largest prime factor
    
    # First, find primes or just iterate and update
    # Efficient way:
    # for i in range(2, limit + 1):
    #     if lpf[i] == 0: # i is prime
    #         for j in range(i, limit + 1, i):
    #             lpf[j] = i
            
    # This is O(N log log N) roughly.
    
    for i in range(2, limit + 1):
        if lpf[i] == 0:
            for j in range(i, limit + 1, i):
                lpf[j] = i
                
    best_n = N_MIN
    best_val = float('inf')
    
    # Iterate through the range
    for n in range(N_MIN, N_MAX + 1):
        val_n = lpf[n]
        val_n1 = lpf[n+1]
        current_val = val_n if val_n > val_n1 else val_n1
        
        if current_val < best_val:
            best_val = current_val
            best_n = n
            
        # Check time
        if time.time() > deadline:
            break
            
    return best_n
