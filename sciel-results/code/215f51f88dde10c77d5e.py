import sys

def construct():
    N_MAX = 1000000
    N_MIN = 10000
    
    # Sieve for smallest prime factor
    spf = list(range(N_MAX + 2))
    for i in range(2, int(N_MAX**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, N_MAX + 2, i):
                if spf[j] == j:
                    spf[j] = i
                    
    def get_lpf(x):
        if x <= 1:
            return 1
        max_p = 1
        while x > 1:
            p = spf[x]
            if p > max_p:
                max_p = p
            while x % p == 0:
                x //= p
        return max_p
        
    best_n = N_MIN
    best_val = get_lpf(N_MIN)
    # We need max(lpf(n), lpf(n+1))
    best_val = max(best_val, get_lpf(N_MIN + 1))
    
    for n in range(N_MIN + 1, N_MAX + 1):
        # Optional early check: if n has a large prime factor, skip?
        # But we need lpf(n+1) too.
        lpf_n = get_lpf(n)
        if lpf_n >= best_val:
            # Even if lpf(n) is already >= best_val, the max will be >= best_val.
            # We want strictly smaller to improve, or equal to keep?
            # The task says "Minimize". If equal, it doesn't matter which n we return, 
            # but the score is seed/value. So we just need the minimum value.
            # If lpf_n >= best_val, then max(lpf_n, lpf_n+1) >= best_val.
            # So we can skip calculating lpf(n+1) if we only care about finding a better (smaller) value.
            # However, if we find an equal value, it doesn't change the score.
            # So we can skip.
            continue
        lpf_np1 = get_lpf(n + 1)
        current_max = lpf_n if lpf_n > lpf_np1 else lpf_np1
        if current_max < best_val:
            best_val = current_max
            best_n = n
            
    return best_n
