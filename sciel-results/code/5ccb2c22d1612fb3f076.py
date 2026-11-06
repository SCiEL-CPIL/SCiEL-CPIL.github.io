import time
import math

def construct():
    deadline = time.time() + 55.0  # safety margin under 90s
    t0 = time.time()

    N = 1000000
    M = N + 1  # need lpf up to N+1

    # Build smallest prime factor table
    spf = list(range(M + 1))
    spf[0] = 0
    spf[1] = 1
    limit = int(math.isqrt(M))
    for i in range(2, limit + 1):
        if spf[i] == i:  # prime
            step = i
            start = i * i
            for j in range(start, M + 1, step):
                if spf[j] == j:
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
