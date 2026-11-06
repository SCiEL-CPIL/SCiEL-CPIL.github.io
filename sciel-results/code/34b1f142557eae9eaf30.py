import math

def construct():
    N = 1000001
    # Sieve for smallest prime factor
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute largest prime factor for each number
    L = [0] * (N + 1)
    L[1] = 1
    for i in range(2, N + 1):
        if spf[i] == i:
            L[i] = i
        else:
            p = spf[i]
            L[i] = max(p, L[i // p])
            
    min_val = float('inf')
    best_n = 10000
    for n in range(10000, 1000000 + 1):
        val = max(L[n], L[n + 1])
        if val < min_val:
            min_val = val
            best_n = n
            
    return best_n
