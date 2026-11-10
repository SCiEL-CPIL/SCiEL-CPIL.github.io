"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 75.0  # safety margin before 90s kill

    def lpf_trial(m):
        if m <= 1:
            return 1
        if m % 2 == 0:
            return m
        r = int(math.isqrt(m)) + 1
        p = 3
        while p <= r:
            if m % p == 0:
                return m
            p += 2
        return m  # prime

    def lpf_of_pair(n, primes):
        a, b = n, n + 1
        for p in primes:
            if p > a:
                break
            if a % p == 0:
                a //= p
            if b % p == 0:
                b //= p
        return max(a, b)

    def sieve(limit):
        s = np.ones(limit + 1, dtype=bool)
        s[0] = s[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if s[i]:
                s[i*i::i] = False
        return np.where(s)[0]

    # ---- Phase 1: exhaustive over [10000, 100000] with full small-prime sieve ----
    small = sieve(1000000)  # primes up to 1e6; used to strip factors from n, n+1
    best_n = 10000
    best_val = lpf_of_pair(10000, small)
    for n in range(10001, 100001):
        v = lpf_of_pair(n, small)
        if v < best_val:
            best_val = v
            best_n = n
        if time.time() > deadline:
            return best_n

    # ---- Phase 2: random sampling over [10000, 1000000] ----
    # Use a smaller prime list for speed on large candidates.
    small2 = small[small <= 100000]
    rng = np.random.default_rng(12345)
    while time.time() < deadline:
        n = int(rng.integers(10000, 1000001))
        v = lpf_of_pair(n, small2)
        if v < best_val:
            best_val = v
            best_n = n
            # refine around the new best with a local window
            lo = max(10000, best_n - 200)
            hi = min(1000000, best_n + 200)
            for m in range(lo, hi + 1):
                if m == best_n:
                    continue
                vm = lpf_of_pair(m, small2)
                if vm < best_val:
                    best_val = vm
                    best_n = m
            if time.time() > deadline:
                return best_n

    return best_n
# EVOLVE-BLOCK-END
