import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit

    N = 123457

    # Compute largest prime factor (LPF) for all numbers up to N using a sieve.
    # lpf[x] = largest prime factor of x.
    lpf = [1] * (N + 1)
    for p in range(2, N + 1):
        if lpf[p] == 1:  # p is prime
            for j in range(p, N + 1, p):
                lpf[j] = p

    best_score = float('inf')
    best_a = 1

    # Exhaustive search: minimize max(lpf[a], lpf[N-a]) over a in [1, N//2]
    half = N // 2 + 1
    for a in range(1, half):
        pa = lpf[a]
        pb = lpf[N - a]
        s = pa if pa > pb else pb
        if s < best_score:
            best_score = s
            best_a = a

    return [best_a, N - best_a]
