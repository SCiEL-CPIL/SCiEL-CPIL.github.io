"""Approximate a ratio inside a divisor-sum fiber."""
import time
import math
from fractions import Fraction
from collections import defaultdict
import numpy as np

# EVOLVE-BLOCK-START

MAX_N = 10000000
DEADLINE_SECONDS = 75

def _compute_sigma(n_max):
    """Compute sigma(n) for all 1..n_max using sieve with NumPy for speed."""
    sigma = np.zeros(n_max + 1, dtype=np.int64)
    for d in range(1, n_max + 1):
        sigma[d::d] += d
    return sigma

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE_SECONDS

    # Compute sigma for all n up to MAX_N
    sigma = _compute_sigma(MAX_N)

    # Group indices by sigma value
    groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        s = int(sigma[n])
        groups[s].append(n)

    best_err = None
    best_a = None
    best_b = None

    for s, lst in groups.items():
        if len(lst) < 2:
            continue
        lst.sort()
        # For each b, find a closest to 1.5*b
        for b in lst:
            target = 1.5 * b
            # Binary search
            lo, hi = 0, len(lst) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if lst[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            for idx in [lo, lo - 1]:
                if 0 <= idx < len(lst):
                    a = lst[idx]
                    if a == b:
                        continue
                    err = abs(Fraction(a, b) - Fraction(3, 2))
                    if best_err is None or err < best_err:
                        best_err = err
                        best_a = a
                        best_b = b

        if time.time() > deadline - 5:
            break

    if best_a is None:
        return [11, 6]
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
