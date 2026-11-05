import time

def construct():
    deadline = time.time() + 85.0  # Safety margin
    
    N = 1000000
    limit = N + 1  # We need lpf for n+1 where n can be N
    
    # Precompute largest prime factor for all numbers up to limit
    # lpf[i] will store the largest prime factor of i
    lpf = [1] * (limit + 1)
    
    # Sieve to find largest prime factor
    for i in range(2, limit + 1):
        if lpf[i] == 1:  # i is prime
            for j in range(i, limit + 1, i):
                lpf[j] = i
                
    best_n = 10000
    best_val = max(lpf[10000], lpf[10001])
    
    # Scan for the optimal n
    for n in range(10001, N + 1):
        if time.time() > deadline:
            break
            
        # The value to minimize is max(lpf(n), lpf(n+1))
        val = max(lpf[n], lpf[n + 1])
        
        if val < best_val:
            best_val = val
            best_n = n
            
    return best_n
