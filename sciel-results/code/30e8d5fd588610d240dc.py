def construct():
    import time
    import math

    N = 123457
    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s kill

    # Sieve for largest prime factor of all numbers up to N
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 1:
            for j in range(i, N + 1, i):
                lpf[j] = i

    # Full scan to find the optimal pair
    best_max_pf = float('inf')
    best_a = 1
    best_b = N - 1

    for a in range(1, N):
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        current_max = pa if pa > pb else pb
        if current_max < best_max_pf:
            best_max_pf = current_max
            best_a = a
            best_b = b
            if best_max_pf == 2:
                break

    return [best_a, best_b]
