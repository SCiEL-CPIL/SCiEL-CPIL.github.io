def construct():
    import math
    
    MAX_N = 1000000
    MIN_N = 10000
    
    # Sieve to find the largest prime factor for each number up to MAX_N
    lpf = [0] * (MAX_N + 2)  # +2 to safely access n+1 when n=MAX_N
    
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if lpf[i] == 0:  # i is a prime
            for j in range(i, MAX_N + 1, i):
                lpf[j] = i
                
    # For numbers that are prime themselves (greater than sqrt(MAX_N)), lpf is still 0.
    # If lpf[x] == 0, x is prime (or 1). Since x >= 10000, x is prime.
    
    min_max_factor = float('inf')
    best_n = MIN_N
    
    for n in range(MIN_N, MAX_N + 1):
        lpf_n = lpf[n] if lpf[n] != 0 else n
        lpf_n1 = lpf[n+1] if lpf[n+1] != 0 else (n+1)
        
        current_max = max(lpf_n, lpf_n1)
        
        if current_max < min_max_factor:
            min_max_factor = current_max
            best_n = n
            
    return best_n
