import time
import math
import numpy as np
from fractions import Fraction
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 80.0

    L = 5_000_000

    # Compute sigma via sieve
    sigma = np.zeros(L + 1, dtype=np.int64)
    for i in range(1, L + 1):
        sigma[i::i] += i

    if time.time() - start_time > deadline - 15:
        return [11, 6]

    # Group numbers by sigma value
    groups = defaultdict(list)
    for n in range(1, L + 1):
        groups[int(sigma[n])].append(n)

    if time.time() - start_time > deadline - 10:
        return [11, 6]

    best_error = Fraction(1, 3)
    best_a, best_b = 11, 6

    # Process groups: for each group, find pairs (a,b) with a/b close to 3/2
    for sig_val, members in groups.items():
        if time.time() - start_time > deadline - 3:
            break
        if len(members) < 2:
            continue

        m = len(members)
        if m > 3000:
            step = max(1, m // 1000)
            indices = range(0, m, step)
        else:
            indices = range(m)

        for idx in indices:
            a_val = members[idx]
            target_b = 2 * a_val / 3.0

            # Binary search for closest to target_b
            lo, hi = 0, m - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if members[mid] < target_b:
                    lo = mid + 1
                else:
                    hi = mid

            for c_idx in [lo, lo - 1, lo + 1, lo - 2, lo + 2]:
                if c_idx < 0 or c_idx >= m:
                    continue
                b_val = members[c_idx]
                if a_val == b_val:
                    continue
                num = abs(2 * a_val - 3 * b_val)
                if num == 0:
                    continue
                den = 2 * b_val
                if num * best_error.denominator < best_error.numerator * den:
                    best_error = Fraction(num, den)
                    best_a, best_b = a_val, b_val

    return [int(best_a), int(best_b)]
