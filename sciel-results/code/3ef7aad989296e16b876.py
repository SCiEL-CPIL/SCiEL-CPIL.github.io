import time
import math

def construct():
    deadline = time.time() + 85
    N = 1000000
    
    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(N + 2))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 2, i):
                if spf[j] == j:
                    spf[j] = i

    # 2. Compute Largest Prime Factor (LPF) for all numbers
    lpf = [0] * (N + 2)
    lpf[1] = 1
    for i in range(2, N + 2):
        p = spf[i]
        # lpf[i] is max of the smallest prime factor and the lpf of the rest
        # Wait, if i = p * k, then lpf(i) = max(p, lpf(k))
        # Since p is the smallest prime factor, k >= 1.
        # If k == 1, lpf(i) = p.
        # If k > 1, lpf(i) = max(p, lpf(k)).
        # Note: lpf(k) is already computed since k < i.
        lpf[i] = max(p, lpf[i // p])

    # 3. Search for n in [10000, N] minimizing max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = float('inf')
    
    # Optimization: We can check if time is running out, but this loop is O(N) with simple ops.
    # 1 million iterations is < 0.1s in Python.
    
    for n in range(10000, N + 1):
        val = lpf[n] if lpf[n] > lpf[n+1] else lpf[n+1]
        if val < best_val:
            best_val = val
            best_n = n

    return best_n
