import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 80  # safety margin within 90s limit

    MAXV = 1000001

    # Build largest-prime-factor array using a sieve.
    # lpf[i] will hold the largest prime factor of i.
    lpf = np.zeros(MAXV + 1, dtype=np.int64)

    # First, find all primes using a simple sieve
    is_prime = np.ones(MAXV + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    limit = int(np.sqrt(MAXV)) + 1
    for i in range(2, limit):
        if is_prime[i]:
            start = i * i
            step = i
            is_prime[start::step] = False

    # Get list of primes
    primes = np.where(is_prime)[0]

    # Now mark LPF for each prime
    for p in primes:
        start_idx = p
        lpf[start_idx::p] = p

    # Now find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    # Use vectorized numpy for speed
    n_values = np.arange(10000, 1000001, dtype=np.int64)
    lpf_n = lpf[n_values]
    lpf_np1 = lpf[n_values + 1]
    max_lpf = np.maximum(lpf_n, lpf_np1)

    # Find the index of the minimum
    min_idx = np.argmin(max_lpf)
    best_n = int(n_values[min_idx])
    best_m = int(max_lpf[min_idx])

    # Verify we're within time budget
    if time.time() > deadline:
        pass  # still return best found

    return best_n
