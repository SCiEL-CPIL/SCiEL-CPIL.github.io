import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85  # Safety margin before 90s limit

    N_MIN = 10000
    N_MAX = 1000000

    # Direct approach: compute largest prime factor for all numbers in [2, N_MAX+1]
    # using a modified sieve. This is O(N log log N) with small constant.
    max_val = N_MAX + 1

    # lpf[i] = largest prime factor of i
    lpf = np.zeros(max_val + 1, dtype=np.int32)
    lpf[1] = 1

    # Use a more efficient sieve: only iterate over primes
    # First, find all primes up to max_val using a simple sieve
    is_prime = np.ones(max_val + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(math.isqrt(max_val)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False

    # Now compute LPF using the primes
    primes = np.where(is_prime)[0]

    for p in primes:
        # Mark multiples of p: lpf[multiple] = max(lpf[multiple], p)
        # Since we process in increasing order, the last prime that divides m is the largest
        # But we need to be careful: we want the LARGEST prime factor
        # So we should iterate and set lpf[p::p] = p, but only if p > current lpf
        # Actually, since primes are in increasing order, the last assignment wins,
        # which gives us the largest prime factor.
        lpf[p::p] = p

    # For each n in [N_MIN, N_MAX], compute max(lpf[n], lpf[n+1])
    n_range = np.arange(N_MIN, N_MAX + 1)
    lpf_n = lpf[n_range]
    lpf_np1 = lpf[n_range + 1]
    combined = np.maximum(lpf_n, lpf_np1)

    # Find the n with minimum combined LPF
    min_idx = np.argmin(combined)
    best_n = int(n_range[min_idx])

    # Check time and potentially refine with local search around best_n
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        # Try to see if we can find something better by checking nearby values
        # The current best is already the global minimum over the full range,
        # so there's nothing better. But let's verify.
        pass

    return best_n
