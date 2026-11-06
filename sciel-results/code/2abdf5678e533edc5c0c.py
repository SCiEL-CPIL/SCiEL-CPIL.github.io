import time
import math

def construct():
    deadline = time.time() + 55.0  # safety margin under 90s
    t0 = time.time()

    N = 1000000
    M = N + 1  # need lpf up to N+1

    # Build smallest prime factor table using bytearray for memory efficiency
    # We only need to store the smallest prime factor, which fits in a uint16 for numbers up to 1e6
    # Actually, for numbers up to 1e6, the smallest prime factor is at most 1e6, which needs 20 bits.
    # So we use an array of ints or a list. Let's stick to list but optimize the sieve.
    
    # A faster way to compute LPF directly without storing SPF for all numbers first?
    # Or just optimize the SPF sieve.
    
    spf = [0] * (M + 1)
    for i in range(2, M + 1):
        if spf[i] == 0:
            spf[i] = i
            if i * i <= M:
                for j in range(i * i, M + 1, i):
                    if spf[j] == 0:
                        spf[j] = i

    # Build largest prime factor table
    lpf = [1] * (M + 1)
    for k in range(2, M + 1):
        p = spf[k]
        rest = k // p
        prev = lpf[rest]
        lpf[k] = p if p >= prev else prev

    # Scan for best n in [10000, N]
    best_n = 10000
    best_L = lpf[10000]
    if lpf[10001] > best_L:
        best_L = lpf[10001]
    
    # Optimization: Pre-compute max_lpf[n] = max(lpf[n], lpf[n+1]) for all n?
    # This saves one comparison in the main loop.
    # Let's try to vectorize or just do it in a tight loop.
    
    for n in range(10001, N + 1):
        a = lpf[n]
        b = lpf[n + 1]
        L = a if a >= b else b
        if L < best_L:
            best_L = L
            best_n = n
        if time.time() > deadline:
            break

    return best_n
