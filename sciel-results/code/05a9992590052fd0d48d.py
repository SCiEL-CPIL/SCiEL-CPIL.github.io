"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import gcd
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 30
    def lcm(a, b):
        return a * b // gcd(a, b)

    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)

    weights = [L // (i + 1) for i in range(n)]

    mid = n // 2
    half1 = weights[:mid]
    half2 = weights[mid:]

    def enumerate_sums(half_weights):
        """Enumerate all sums with coefficients in {-1,0,1} using set for dedup."""
        sums = {0}
        for w in half_weights:
            new_sums = set()
            for s in sums:
                new_sums.add(s - w)
                new_sums.add(s)
                new_sums.add(s + w)
            sums = new_sums
        return list(sums)

    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    sums1 = enumerate_sums(half1)
    sums1.sort()
    sums1_arr = np.array(sums1, dtype=np.int64)

    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    sums2 = enumerate_sums(half2)

    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    best_abs = None
    best_s1 = None
    best_s2 = None
    found_min = False

    n1 = len(sums1)

    # Process sums2 in chunks with numpy for speed
    chunk_size = 100000
    for chunk_start in range(0, len(sums2), chunk_size):
        if time.time() > deadline:
            break
        if found_min:
            break

        chunk = sums2[chunk_start:chunk_start + chunk_size]
        targets = np.array([-s for s in chunk], dtype=np.int64)

        # Vectorized binary search
        positions = np.searchsorted(sums1_arr, targets, side='left')

        # For each target, check the closest neighbors
        for i in range(len(targets)):
            pos = positions[i]
            s2_val = chunk[i]
            candidates = []
            if pos > 0:
                candidates.append(pos - 1)
            if pos < n1:
                candidates.append(pos)
            if pos + 1 < n1:
                candidates.append(pos + 1)

            for idx in candidates:
                s1_val = sums1[idx]
                total = s1_val + s2_val
                if total != 0:
                    abs_total = abs(total)
                    if best_abs is None or abs_total < best_abs:
                        best_abs = abs_total
                        best_s1 = s1_val
                        best_s2 = s2_val
                        if best_abs == 1:
                            found_min = True
                            break
            if found_min:
                break

    if best_abs is None:
        return [0] * 28 + [1, -1]

    def decode_sum(half_weights, target_sum):
        """Decode a target sum back to coefficients using DFS."""
        n_h = len(half_weights)
        coeffs = [0] * n_h

        def backtrack(pos, remaining):
            if pos == n_h:
                return remaining == 0
            w = half_weights[pos]
            # Try 0 first (most common), then 1, then -1
            for c in (0, 1, -1):
                if backtrack(pos + 1, remaining - c * w):
                    coeffs[pos] = c
                    return True
            return False

        if backtrack(0, target_sum):
            return coeffs
        return None

    coeffs1 = decode_sum(half1, best_s1)
    coeffs2 = decode_sum(half2, best_s2)

    if coeffs1 is None or coeffs2 is None:
        return [0] * 28 + [1, -1]

    result = coeffs1 + coeffs2

    # Verify
    total = 0
    for i in range(n):
        total += result[i] * weights[i]

    if total == 0:
        return [0] * 28 + [1, -1]

    # Local search polish: try to reduce |sum| further by flipping single coefficients
    if time.time() < deadline - 5.0:
        current_sum = total
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(n):
                if time.time() > deadline - 5.0:
                    break
                for delta in (-2, -1, 1, 2):
                    new_c = result[i] + delta
                    if new_c in (-1, 0, 1):
                        new_sum = current_sum + delta * weights[i]
                        if new_sum != 0 and abs(new_sum) < abs(current_sum):
                            result[i] = new_c
                            current_sum = new_sum
                            improved = True
                            break

    # Final verification
    final_sum = 0
    for i in range(n):
        final_sum += result[i] * weights[i]

    if final_sum == 0:
        return [0] * 28 + [1, -1]

    return result

# EVOLVE-BLOCK-END
