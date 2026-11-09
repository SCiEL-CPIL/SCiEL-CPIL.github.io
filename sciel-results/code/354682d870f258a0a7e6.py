"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import product

def _lcm(a, b):
    return a * b // gcd(a, b)

def _build_lcm():
    L = 1
    for i in range(1, 31):
        L = _lcm(L, i)
    return L

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill

    L = _build_lcm()
    # weights[i] for coefficient at position i (1-indexed denominator i)
    # sum_{i=1}^{30} c[i] * (L // i)
    weights = [0] + [L // i for i in range(1, 31)]  # index 1..30

    # We want to minimize |sum_{i=1}^{30} c[i] * w[i]| subject to sum != 0
    # and c[i] in {-1, 0, 1}

    # Meet-in-the-middle: split into first 15 and last 15
    # First half: indices 1..15, second half: indices 16..30
    # Enumerate all 3^15 = 14348907 sums for each half

    # Build list of (sum_value, coeff_tuple) for first half
    half1_weights = weights[1:16]  # w[1]..w[15]
    half2_weights = weights[16:31]  # w[16]..w[30]

    # Generate all sums for first half
    sums1 = []
    # Use iterative generation to avoid deep recursion
    # 3^15 = 14348907, manageable
    n1 = len(half1_weights)
    n2 = len(half2_weights)

    # Precompute powers of 3 for indexing
    pow3 = [1]
    for i in range(30):
        pow3.append(pow3[-1] * 3)

    def gen_sums(weights_list):
        n = len(weights_list)
        total = pow3[n]
        sums = []
        for idx in range(total):
            s = 0
            temp = idx
            for i in range(n):
                digit = temp % 3
                temp //= 3
                if digit == 1:
                    s += weights_list[i]
                elif digit == 2:
                    s -= weights_list[i]
                # digit 0: add nothing
            sums.append(s)
        return sums

    # Generate sums for both halves
    sums1 = gen_sums(half1_weights)
    sums2 = gen_sums(half2_weights)

    # Sort sums2 to enable binary search
    sums2_sorted = sorted(sums2)

    # For each sum in sums1, find the sum in sums2 that makes total closest to 0 but not 0
    # We want min |s1 + s2| where s1 + s2 != 0
    # Binary search for -s1 in sums2_sorted

    best_abs = None
    best_s1 = None
    best_s2 = None

    import bisect

    for s1 in sums1:
        # We want s2 such that s1 + s2 is close to 0 but not 0
        # Target: -s1
        # Find insertion point
        target = -s1
        pos = bisect.bisect_left(sums2_sorted, target)

        # Check candidates around pos
        candidates = []
        if pos < len(sums2_sorted):
            candidates.append(sums2_sorted[pos])
        if pos > 0:
            candidates.append(sums2_sorted[pos - 1])
        # Also check pos+1 in case
        if pos + 1 < len(sums2_sorted):
            candidates.append(sums2_sorted[pos + 1])

        for s2 in candidates:
            total = s1 + s2
            if total == 0:
                continue
            abs_total = abs(total)
            if best_abs is None or abs_total < best_abs:
                best_abs = abs_total
                best_s1 = s1
                best_s2 = s2

    # Check time
    if time.time() > deadline:
        # Should not happen, but safety
        pass

    # Now we need to recover the actual coefficients
    # best_s1 corresponds to some coefficient pattern in first half
    # best_s2 corresponds to some coefficient pattern in second half
    # We need to find which pattern gives best_s1 and best_s2

    def find_pattern(sums_list, weights_list, target_sum):
        n = len(weights_list)
        total = pow3[n]
        for idx in range(total):
            s = 0
            temp = idx
            coeffs = [0] * n
            for i in range(n):
                digit = temp % 3
                temp //= 3
                if digit == 1:
                    s += weights_list[i]
                    coeffs[i] = 1
                elif digit == 2:
                    s -= weights_list[i]
                    coeffs[i] = -1
                else:
                    coeffs[i] = 0
            if s == target_sum:
                return coeffs
        return None

    # This could be slow if done naively, but we just need one match
    # Let's use a more efficient approach: rebuild the mapping

    # Actually, let's just find the first occurrence
    coeffs1 = find_pattern(sums1, half1_weights, best_s1)
    coeffs2 = find_pattern(sums2, half2_weights, best_s2)

    if coeffs1 is None or coeffs2 is None:
        # Fallback: return the seed answer
        return [0] * 28 + [1, -1]

    # Combine: coeffs1 is for indices 1-15, coeffs2 for indices 16-30
    # The output should be coefficient[i-1] for i=1..30
    # So output[0] = c[1], output[1] = c[2], ..., output[29] = c[30]
    result = coeffs1 + coeffs2

    # Verify
    total = sum(result[i] * (L // (i+1)) for i in range(30))
    if total == 0:
        # Should not happen, but fallback
        return [0] * 28 + [1, -1]

    return result

# EVOLVE-BLOCK-END
