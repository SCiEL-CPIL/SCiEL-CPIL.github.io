import time
from fractions import Fraction
import bisect

def construct():
    deadline = time.time() + 55  # 55s deadline for safety

    denoms = list(range(2, 49))
    N = len(denoms)

    # Split into two halves
    mid = N // 2  # 23
    half1 = denoms[:mid]  # 23 items: [2, 3, ..., 24]
    half2 = denoms[mid:]  # 24 items: [25, 26, ..., 48]

    n1 = len(half1)
    n2 = len(half2)

    # Generate all subset sums for half1 (float)
    sums1 = [0.0] * (1 << n1)
    for i in range(n1):
        val = 1.0 / half1[i]
        for mask in range(1 << i):
            sums1[mask | (1 << i)] = sums1[mask] + val

    # Generate all subset sums for half2 (float)
    sums2 = [0.0] * (1 << n2)
    for i in range(n2):
        val = 1.0 / half2[i]
        for mask in range(1 << i):
            sums2[mask | (1 << i)] = sums2[mask] + val

    # Sort half2 sums and keep masks
    pairs2 = list(enumerate(sums2))
    pairs2.sort(key=lambda x: x[1])
    sorted_vals2 = [p[1] for p in pairs2]

    # Precompute exact Fraction sums for half2 to speed up evaluation
    fracs2 = [Fraction(1, d) for d in half2]
    exact_sums2 = [Fraction(0)] * (1 << n2)
    for i in range(n2):
        f = fracs2[i]
        for mask in range(1 << i):
            exact_sums2[mask | (1 << i)] = exact_sums2[mask] + f

    # Precompute exact Fraction sums for half1
    fracs1 = [Fraction(1, d) for d in half1]
    exact_sums1 = [Fraction(0)] * (1 << n1)
    for i in range(n1):
        f = fracs1[i]
        for mask in range(1 << i):
            exact_sums1[mask | (1 << i)] = exact_sums1[mask] + f

    best_sum_exact = Fraction(0)
    best_mask1 = 0
    best_mask2 = 0
    best_float_sum = 0.0

    # Sort half1 masks by float sum descending to find good candidates early
    mask_order1 = sorted(range(1 << n1), key=lambda m: sums1[m], reverse=True)

    for m1 in mask_order1:
        s1_float = sums1[m1]
        if s1_float >= 1.0:
            continue

        target = 1.0 - s1_float

        # Find the largest value in sorted_vals2 strictly less than target
        idx = bisect.bisect_left(sorted_vals2, target) - 1

        if idx < 0:
            continue

        m2 = pairs2[idx][0]
        s2_float = sorted_vals2[idx]
        combined_float = s1_float + s2_float

        if combined_float > best_float_sum:
            # Compute exact sum
            s_total = exact_sums1[m1] + exact_sums2[m2]

            if s_total < 1:
                if s_total > best_sum_exact:
                    best_sum_exact = s_total
                    best_mask1 = m1
                    best_mask2 = m2
                    best_float_sum = combined_float

        if time.time() > deadline:
            break

    # Construct indices
    indices = []
    for i in range(n1):
        if best_mask1 & (1 << i):
            indices.append(i)

    for i in range(n2):
        if best_mask2 & (1 << i):
            indices.append(mid + i)

    if not indices:
        return [0]  # Fallback: denominator 2, sum 0.5 < 1

    return indices
