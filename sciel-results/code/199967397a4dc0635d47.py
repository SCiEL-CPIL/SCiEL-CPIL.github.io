import time
import math

def construct():
    deadline = time.time() + 85.0
    TARGET = 123457

    # Precompute largest prime factor for all n up to TARGET using a sieve
    # This is O(N log log N) and very fast for N=123457
    lpf = [0] * (TARGET + 1)
    lpf[1] = 1
    for i in range(2, TARGET + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, TARGET + 1, i):
                lpf[j] = i

    best_val = float('inf')
    best_pair = (1, TARGET - 1)

    # Iterate through all possible a
    # Since N is odd, one of a, b must be even and one odd.
    # The minimum possible max(P(a), P(b)) is at least 3 (since one is odd > 1).
    # If we find 3, it's likely optimal.
    for a in range(1, TARGET):
        if time.time() > deadline - 1.0:
            break
        b = TARGET - a
        pa = lpf[a]
        pb = lpf[b]
        val = pa if pa > pb else pb
        if val < best_val:
            best_val = val
            best_pair = (a, b)
            # Early exit if we hit a very low value
            if best_val <= 3:
                break

    a, b = best_pair
    if time.time() > deadline:
        return [1, TARGET - 1]
    return [a, b]
