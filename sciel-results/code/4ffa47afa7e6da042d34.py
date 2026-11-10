import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin of 5 seconds
    
    N = 1000000
    # We need lpf for n and n+1, so up to N+1
    max_val = N + 1
    lpf = [0] * (max_val + 2)
    
    # Sieve to find largest prime factor
    # i is the potential prime factor
    for i in range(2, max_val + 2):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, max_val + 2, i):
                lpf[j] = i
                
    if time.time() > deadline:
        return 10000

    min_lpf_val = float('inf')
    best_n = 10000
    
    # Search for n in [10000, 1000000]
    # We want to minimize max(lpf[n], lpf[n+1])
    for n in range(10000, N + 1):
        # Check deadline
        if time.time() > deadline:
            break
            
        current_max_lpf = lpf[n]
        next_max_lpf = lpf[n+1]
        
        if next_max_lpf > current_max_lpf:
            val = next_max_lpf
        else:
            val = current_max_lpf
            
        if val < min_lpf_val:
            min_lpf_val = val
            best_n = n
            
    return best_n
