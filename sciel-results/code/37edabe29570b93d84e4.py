import time
import numpy as np

def construct():
    N = 123457
    deadline = time.time() + 85  # safety margin before 90s limit

    # Precompute largest prime factor (LPF) for all numbers up to N
    # Using a sieve-like approach with NumPy for speed
    lpf = np.zeros(N + 1, dtype=np.int64)
    lpf[1] = 1

    for p in range(2, N + 1):
        if lpf[p] == 0:  # p is prime
            # Mark all multiples of p: lpf[j] = p (largest prime factor so far)
            lpf[p::p] = p

    # O(N) scan: for each a in [1, N//2], compute max(lpf[a], lpf[N-a])
    # and track the minimum. This finds the optimal pair in one pass.
    best_score = float('inf')
    best_a = 1
    best_b = N - 1

    mid = N // 2
    for a in range(1, mid + 1):
        if time.time() > deadline:
            break
        b = N - a
        score = max(lpf[a], lpf[b])
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b

    return [int(best_a), int(best_b)]
