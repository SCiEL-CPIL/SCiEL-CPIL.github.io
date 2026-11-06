import math
import time
from functools import reduce
from collections import deque
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = 30
    # Compute LCM of 1..30
    H = reduce(math.lcm, range(1, n + 1))
    W = [H // i for i in range(1, n + 1)]
    
    # Meet-in-the-middle: split into two halves
    mid = 15
    
    # Generate all possible sums for first half (indices 0..mid-1)
    # We only store the sum values, not the full coefficient vectors
    sums_a = {0}
    for i in range(mid):
        w = W[i]
        new_sums = set()
        for s in sums_a:
            new_sums.add(s + w)
            new_sums.add(s)
            new_sums.add(s - w)
        sums_a = new_sums
        if time.time() > deadline:
            return _fallback(W, n)
    
    # Generate all possible sums for second half (indices mid..n-1)
    sums_b = {0}
    for i in range(n - mid):
        w = W[mid + i]
        new_sums = set()
        for s in sums_b:
            new_sums.add(s + w)
            new_sums.add(s)
            new_sums.add(s - w)
        sums_b = new_sums
        if time.time() > deadline:
            return _fallback(W, n)
    
    # Convert B sums to sorted list for binary search
    import bisect
    arr_b = sorted(sums_b)
    b_min = arr_b[0]
    b_max = arr_b[-1]
    
    best_val = None
    best_a = None
    best_b = None
    
    # Sort A sums by absolute value to find small |a+b| faster
    a_list = sorted(sums_a, key=lambda a: abs(a))
    
    for a in a_list:
        if time.time() > deadline:
            break
        
        if best_val is not None and best_val == 1:
            break
        
        target = -a
        if target < b_min:
            val = a + b_min
            if val != 0:
                abs_val = abs(val)
                if best_val is None or abs_val < best_val:
                    best_val = abs_val
                    best_a = a
                    best_b = b_min
        elif target > b_max:
            val = a + b_max
            if val != 0:
                abs_val = abs(val)
                if best_val is None or abs_val < best_val:
                    best_val = abs_val
                    best_a = a
                    best_b = b_max
        else:
            idx = bisect.bisect_left(arr_b, target)
            for candidate_idx in [idx, idx - 1]:
                if 0 <= candidate_idx < len(arr_b):
                    b = arr_b[candidate_idx]
                    val = a + b
                    if val != 0:
                        abs_val = abs(val)
                        if best_val is None or abs_val < best_val:
                            best_val = abs_val
                            best_a = a
                            best_b = b
                    if best_val == 1:
                        break
            if best_val == 1:
                break
    
    if best_a is not None and best_b is not None:
        # Reconstruct coefficients for the first half
        coeffs_a = _reconstruct_half(W, mid, best_a)
        # Reconstruct coefficients for the second half
        coeffs_b = _reconstruct_half(W, n - mid, best_b, offset=mid)
        final_coeffs = coeffs_a + coeffs_b
        total = sum(c * W[i] for i, c in enumerate(final_coeffs))
        if total != 0:
            return final_coeffs
    
    return _fallback(W, n)

def _reconstruct_half(W, length, target_sum, offset=0):
    """Reconstruct coefficients for a half given the target sum."""
    coeffs = [0] * length
    remaining = target_sum
    
    for i in range(length - 1, -1, -1):
        w = W[offset + i]
        # Try each coefficient value
        found = False
        for c in [1, 0, -1]:
            new_remaining = remaining - c * w
            # Check if new_remaining is achievable with remaining indices
            # Simple check: |new_remaining| <= sum of remaining weights
            remaining_weights = sum(W[offset + j] for j in range(i))
            if abs(new_remaining) <= remaining_weights:
                coeffs[i] = c
                remaining = new_remaining
                found = True
                break
        if not found:
            # Fallback: try all values
            for c in [1, 0, -1]:
                new_remaining = remaining - c * w
                remaining_weights = sum(W[offset + j] for j in range(i))
                if abs(new_remaining) <= remaining_weights:
                    coeffs[i] = c
                    remaining = new_remaining
                    found = True
                    break
            if not found:
                # Should not happen, but handle gracefully
                coeffs[i] = 0
    
    return coeffs

def _fallback(W, n):
    """Greedy fallback: construct a small nonzero sum."""
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
