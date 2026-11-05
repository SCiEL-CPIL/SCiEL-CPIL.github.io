import time
from math import gcd
from functools import reduce
import bisect

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit

    n = 30
    # LCM of 1..30
    L = reduce(lambda a, b: a * b // gcd(a, b), range(1, n + 1), 1)
    weights = [L // i for i in range(1, n + 1)]

    mid = n // 2  # 15
    w1 = weights[:mid]
    w2 = weights[mid:]

    def gen_sums_fast(weights):
        m = len(weights)
        size = 3 ** m
        sums = [0] * size
        for k in range(m):
            w = weights[k]
            p3 = 3 ** k
            for base in range(0, size, p3 * 3):
                for offset in range(p3):
                    s = sums[base + offset]
                    sums[base + offset] = s - w
                    sums[base + p3 + offset] = s
                    sums[base + 2 * p3 + offset] = s + w
        return sums

    sums1 = gen_sums_fast(w1)
    sums2 = gen_sums_fast(w2)

    if time.time() - start_time > deadline:
        return [0] * 29 + [1]

    # Create sorted lists with original indices
    size1 = len(sums1)
    size2 = len(sums2)

    # Sort sums1
    order1 = sorted(range(size1), key=lambda i: sums1[i])
    sums1_sorted = [sums1[order1[i]] for i in range(size1)]

    # Sort sums2
    order2 = sorted(range(size2), key=lambda i: sums2[i])
    sums2_sorted = [sums2[order2[i]] for i in range(size2)]

    if time.time() - start_time > deadline:
        return [0] * 29 + [1]

    # For each s2, binary search for -s2 in sums1_sorted
    best_abs = float('inf')
    best_diff = None
    best_idx1 = 0
    best_idx2 = 0

    for i2 in range(size2):
        s2 = sums2_sorted[i2]
        target = -s2

        # Binary search in sums1_sorted
        pos = bisect.bisect_left(sums1_sorted, target)

        # Check pos and pos-1
        for p in [pos, pos - 1]:
            if 0 <= p < size1:
                s1 = sums1_sorted[p]
                diff = s1 + s2
                if diff == 0:
                    continue
                ad = abs(diff)
                if ad < best_abs:
                    best_abs = ad
                    best_diff = diff
                    best_idx1 = order1[p]
                    best_idx2 = order2[i2]
                    if ad == 1:
                        break

        if best_abs == 1:
            break

        # Time check every 1000 iterations
        if i2 % 1000 == 0 and time.time() - start_time > deadline - 5.0:
            break

    if best_diff is None:
        return [0] * 29 + [1]

    # Reconstruct coefficients from original indices
    coeffs = [0] * n

    t = best_idx1
    for i in range(mid):
        d = t % 3
        t //= 3
        coeffs[i] = d - 1

    t = best_idx2
    for i in range(n - mid):
        d = t % 3
        t //= 3
        coeffs[mid + i] = d - 1

    return coeffs
