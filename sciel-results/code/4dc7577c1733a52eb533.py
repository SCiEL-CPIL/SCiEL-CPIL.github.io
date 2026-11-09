import time
import numpy as np
from collections import Counter

def construct():
    start = time.time()
    deadline = start + 82  # safety margin before 90s limit

    N = 1000000  # max n
    WINDOW = 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    MAX_VAL = N + WINDOW

    # Compute sp[m] = complete factor of m composed of primes < 24
    # Vectorized sieve: for each prime power, multiply all multiples
    sp = np.ones(MAX_VAL + 1, dtype=np.int64)
    for p in PRIMES:
        power = p
        while power <= MAX_VAL:
            sp[power::power] *= p
            power *= p

    # Convert to list for fast Python slicing
    sp_list = sp.tolist()

    best_n = 1000
    best_distinct = float('inf')

    # Phase 1: Coarse scan with step 64 to quickly find promising regions
    # This covers ~15625 candidates, each evaluation is O(24) set build
    step1 = 64
    for n in range(1000, N + 1, step1):
        if time.time() > deadline - 20:
            break
        factors = set(sp_list[n + 1: n + 25])
        distinct_count = len(factors)
        if distinct_count < best_distinct:
            best_distinct = distinct_count
            best_n = n

    # Phase 2: Medium scan with step 8 in the top regions found
    # Find the best region from phase 1 and scan around it
    if best_distinct <= 15:
        # Scan a wider range around the best with step 8
        lo = max(1000, best_n - 2000)
        hi = min(N, best_n + 2000)
        for n in range(lo, hi + 1, 8):
            if time.time() > deadline - 10:
                break
            factors = set(sp_list[n + 1: n + 25])
            distinct_count = len(factors)
            if distinct_count < best_distinct:
                best_distinct = distinct_count
                best_n = n

    # Phase 3: Fine scan with step 1 around the best candidate
    if best_distinct <= 12:
        lo = max(1000, best_n - 200)
        hi = min(N, best_n + 200)
        for n in range(lo, hi + 1):
            if time.time() > deadline - 3:
                break
            factors = set(sp_list[n + 1: n + 25])
            distinct_count = len(factors)
            if distinct_count < best_distinct:
                best_distinct = distinct_count
                best_n = n

    # Phase 4: Final local refinement
    if time.time() < deadline - 2:
        lo = max(1000, best_n - 50)
        hi = min(N, best_n + 50)
        for n in range(lo, hi + 1):
            if time.time() > deadline - 1:
                break
            factors = set(sp_list[n + 1: n + 25])
            distinct_count = len(factors)
            if distinct_count < best_distinct:
                best_distinct = distinct_count
                best_n = n

    # Phase 5: If still not great, do a broader medium scan
    if best_distinct > 10 and time.time() < deadline - 5:
        step2 = 16
        for n in range(1000, N + 1, step2):
            if time.time() > deadline - 2:
                break
            factors = set(sp_list[n + 1: n + 25])
            distinct_count = len(factors)
            if distinct_count < best_distinct:
                best_distinct = distinct_count
                best_n = n
                # Quick local refinement
                if time.time() < deadline - 3:
                    lo2 = max(1000, best_n - 20)
                    hi2 = min(N, best_n + 20)
                    for m in range(lo2, hi2 + 1):
                        if time.time() > deadline - 2:
                            break
                        v2 = len(set(sp_list[m + 1: m + 25]))
                        if v2 < best_distinct:
                            best_distinct = v2
                            best_n = m

    return best_n
