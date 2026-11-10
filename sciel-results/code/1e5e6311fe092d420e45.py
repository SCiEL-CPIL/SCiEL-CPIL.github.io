import time
from math import gcd
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    DEADLINE = 80.0  # wall-clock safety margin (evaluator kills at 90s)
    n = 30
    k = 15
    left_terms = list(range(1, k + 1))
    right_terms = list(range(k + 1, n + 1))

    # ---- Build right half (deterministic, fast) ----
    # value = sum over j of coeff_j / (k+1+j)  where j in 0..k-1
    # represent as Fraction
    right = {}
    for mask in range(1 << k):
        s = Fraction(0, 1)
        m = mask
        j = 0
        while m:
            if m & 1:
                s += Fraction(1, right_terms[j])
            m >>= 1
            j += 1
        right[s] = mask

    # ---- Build left half (deterministic, fast) ----
    # value = sum over i of coeff_i / (i+1)  where i in 0..k-1
    left = {}
    for mask in range(1 << k):
        s = Fraction(0, 1)
        m = mask
        i = 0
        while m:
            if m & 1:
                s += Fraction(1, left_terms[i])
            m >>= 1
            i += 1
        left[s] = mask

    # ---- Search for minimal |total| where total = left_val + right_val, total != 0 ----
    # We want to minimize |l + r| over l in left, r in right, l + r != 0.
    # Sort right values; for each left value l, find r closest to -l but with l+r != 0.
    import bisect
    sorted_right_vals = sorted(right.keys())
    best_abs = None
    best_pair = None

    for l in left.keys():
        target = -l
        # find insertion point
        idx = bisect.bisect_left(sorted_right_vals, target)
        candidates = []
        for pos in (idx - 1, idx, idx + 1):
            if 0 <= pos < len(sorted_right_vals):
                candidates.append(sorted_right_vals[pos])
        # dedupe
        seen = set()
        for r in candidates:
            if r in seen:
                continue
            seen.add(r)
            total = l + r
            if total == 0:
                continue
            a = abs(total)
            if best_abs is None or a < best_abs:
                best_abs = a
                best_pair = (l, r)
        # time guard
        if time.time() > DEADLINE - 5.0:
            break

    if best_pair is None:
        # fallback: seed-like answer
        return [0] * 28 + [1, -1]

    l_val, r_val = best_pair
    l_mask = left[l_val]
    r_mask = right[r_val]

    # Reconstruct coefficients
    coeffs = [0] * n
    # left: index i (0..k-1) corresponds to term 1/(i+1)
    for i in range(k):
        if (l_mask >> i) & 1:
            coeffs[i] = 1
    # right: index j (0..k-1) corresponds to term 1/(k+1+j)
    for j in range(k):
        if (r_mask >> j) & 1:
            coeffs[k + j] = 1

    # Verify nonzero
    total = Fraction(0, 1)
    for i in range(n):
        if coeffs[i] != 0:
            total += Fraction(coeffs[i], i + 1)
    if total == 0:
        return [0] * 28 + [1, -1]

    return coeffs
# EVOLVE-BLOCK-END
