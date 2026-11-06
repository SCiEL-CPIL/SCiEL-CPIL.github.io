import math
import time
import numpy as np
from functools import reduce

def _generate_sums(weights):
    """Generate all possible signed sums using vectorized numpy."""
    n = len(weights)
    total = 3 ** n
    sums = np.zeros(total, dtype=np.int64)
    for i, w in enumerate(weights):
        block = 3 ** i
        pattern = np.array([-1, 0, 1], dtype=np.int64) * w
        reps = total // (3 * block)
        sums += np.tile(pattern, reps)
    return sums

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 30
    H = reduce(math.lcm, range(1, n + 1))
    W = [H // i for i in range(1, n + 1)]
    
    mid = 15
    weights_a = W[:mid]
    weights_b = W[mid:]
    
    sums_a = _generate_sums(weights_a)
    if time.time() > deadline:
        return _fallback(W, n)
    
    sums_b = _generate_sums(weights_b)
    if time.time() > deadline:
        return _fallback(W, n)
    
    # Sort
    sums_a.sort()
    sums_b.sort()
    
    # Two-pointer search for min |a+b| != 0
    best_val = None
    best_a = None
    best_b = None
    i, j = 0, len(sums_b) - 1
    n_a, n_b = len(sums_a), len(sums_b)
    
    while i < n_a and j >= 0:
        if time.time() > deadline:
            break
        s = sums_a[i] + sums_b[j]
        if s == 0:
            # Check neighbors for nonzero
            if i + 1 < n_a:
                val = sums_a[i + 1] + sums_b[j]
                if val != 0:
                    av = abs(val)
                    if best_val is None or av < best_val:
                        best_val = av
                        best_a = sums_a[i + 1]
                        best_b = sums_b[j]
                        if best_val == 1:
                            break
            if j - 1 >= 0:
                val = sums_a[i] + sums_b[j - 1]
                if val != 0:
                    av = abs(val)
                    if best_val is None or av < best_val:
                        best_val = av
                        best_a = sums_a[i]
                        best_b = sums_b[j - 1]
                        if best_val == 1:
                            break
            i += 1
            j -= 1
        elif s > 0:
            av = abs(s)
            if best_val is None or av < best_val:
                best_val = av
                best_a = sums_a[i]
                best_b = sums_b[j]
                if best_val == 1:
                    break
            j -= 1
        else:
            av = abs(s)
            if best_val is None or av < best_val:
                best_val = av
                best_a = sums_a[i]
                best_b = sums_b[j]
                if best_val == 1:
                    break
            i += 1
    
    if best_a is not None and best_b is not None:
        coeffs_a = _find_coeffs(int(best_a), 0, mid, W)
        coeffs_b = _find_coeffs(int(best_b), mid, n, W)
        if coeffs_a is not None and coeffs_b is not None:
            final_coeffs = coeffs_a + coeffs_b
            total = sum(c * W[i] for i, c in enumerate(final_coeffs))
            if total != 0:
                return final_coeffs
    
    return _fallback(W, n)

def _find_coeffs(target, start, end, W):
    if start == end:
        return [] if target == 0 else None
    max_rem = sum(W[start:end])
    if abs(target) > max_rem:
        return None
    w = W[start]
    for c in [-1, 0, 1]:
        res = _find_coeffs(target - c * w, start + 1, end, W)
        if res is not None:
            return [c] + res
    return None

def _fallback(W, n):
    coeffs = [0] * n
    curr = 0
    for i in range(n - 1, -1, -1):
        w = W[i]
        best_c = 0
        best_val = curr
        for c in [-1, 0, 1]:
            val = curr + c * w
            if abs(val) < abs(best_val):
                best_val = val
                best_c = c
        coeffs[i] = best_c
        curr = best_val
    if curr == 0:
        min_idx = min(range(n), key=lambda i: W[i])
        coeffs[min_idx] = 1
        curr = sum(c * W[i] for i, c in enumerate(coeffs))
    return coeffs
