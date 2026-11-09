def construct():
    import math
    
    MAX_N = 1000000
    MIN_N = 10000
    
    # Sieve of Eratosthenes to find Smallest Prime Factor (SPF)
    spf = list(range(MAX_N + 2))
    for i in range(2, int(math.isqrt(MAX_N + 1)) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_N + 2, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_lpf(x):
        max_p = 1
        while x > 1:
            p = spf[x]
            if p > max_p:
                max_p = p
            while x % p == 0:
                x //= p
        return max_p
    
    best_n = MIN_N
    best_lpf = float('inf')
    
    for n in range(MIN_N, MAX_N + 1):
        lpf_n = get_lpf(n)
        lpf_np1 = get_lpf(n + 1)
        current_max_lpf = lpf_n if lpf_n > lpf_np1 else lpf_np1
        
        if current_max_lpf < best_lpf:
            best_lpf = current_max_lpf
            best_n = n
            
            # Minimum possible LPF for n >= 10000 is 5
            # (3-smooth consecutive pairs are only 8,9)
            if best_lpf == 5:
                break
                
    return best_n
