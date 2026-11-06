import math
import time
from functools import reduce
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = 30
    # Compute LCM of 1..30
    H = reduce(math.lcm, range(1, n + 1))
    W = [H // i for i in range(1, n + 1)]
    
    # Split into two halves
    mid = 15
    
    # Generate all possible sums for first half (indices 0..14)
    # Using list for faster generation, then sort/unique
    sums_a = [0]
    for i in range(mid):
        w = W[i]
        # Create new list with 3x size
        # Optimization: pre-allocate or just extend
        new_sums = []
        # Using list comprehension or loop
        # Loop is often faster in Python for complex ops, but list comp is C-speed
        # However, we are adding 3 items per existing item.
        # new_sums.extend([s + w, s, s - w] for s in sums_a) is slow due to generator overhead?
        # Let's use a standard loop with local variables for speed
        append = new_sums.append
        for s in sums_a:
            append(s + w)
            append(s)
            append(s - w)
        sums_a = new_sums
        
    if time.time() > deadline:
        return _fallback(W, n)
        
    # Sort and unique A
    sums_a.sort()
    # In-place unique
    if sums_a:
        unique_a = [sums_a[0]]
        last = sums_a[0]
        for s in sums_a[1:]:
            if s != last:
                unique_a.append(s)
                last = s
        sums_a = unique_a
    else:
        sums_a = []

    # Generate all possible sums for second half (indices 15..29)
    sums_b = [0]
    for i in range(mid, n):
        w = W[i]
        new_sums = []
        append = new_sums.append
        for s in sums_b:
            append(s + w)
            append(s)
            append(s - w)
        sums_b = new_sums
        
    if time.time() > deadline:
        return _fallback(W, n)

    # Sort and unique B
    sums_b.sort()
    if sums_b:
        unique_b = [sums_b[0]]
        last = sums_b[0]
        for s in sums_b[1:]:
            if s != last:
                unique_b.append(s)
                last = s
        sums_b = unique_b
    else:
        sums_b = []

    # Now we have sorted unique lists sums_a and sums_b
    # We want to find min |a + b| != 0
    
    # To speed up search, we can iterate over the smaller list?
    # Both are approx same size.
    # Iterate over sums_a, binary search in sums_b.
    
    best_val = None
    best_a = None
    best_b = None
    
    # Optional: Sort sums_a by absolute value to find small sums earlier?
    # If we find 1, we break.
    # Small sums are more likely to be found if we check a's that are close to -b's.
    # But binary search is O(log N).
    # If we sort sums_a by abs value, we might find a solution with small |a+b| earlier?
    # Not necessarily. a could be large, but b cancels it.
    # However, if we find best_val == 1, we stop.
    # So finding 1 quickly is key.
    
    # Let's just iterate.
    b_min = sums_b[0]
    b_max = sums_b[-1]
    
    # To make it faster, we can use two pointers if we sort sums_a as well?
    # sums_a is already sorted.
    # We can use two pointers to find pairs with sum closest to 0.
    # But we need min |a+b| != 0.
    # Two pointers:
    # i = 0 (smallest a), j = len(sums_b) - 1 (largest b)
    # If a[i] + b[j] > 0, decrement j.
    # If a[i] + b[j] < 0, increment i.
    # If a[i] + b[j] == 0, check neighbors.
    # This is O(N + M). Much faster than N log M!
    
    i = 0
    j = len(sums_b) - 1
    n_a = len(sums_a)
    n_b = len(sums_b)
    
    while i < n_a and j >= 0:
        if time.time() > deadline:
            break
            
        s = sums_a[i] + sums_b[j]
        if s == 0:
            # We found a zero sum. We need the nearest non-zero sum.
            # Check s + delta for small deltas?
            # Or just check neighbors of i and j.
            # The candidates are (i, j-1), (i+1, j), (i+1, j-1), etc.
            # Since lists are sorted, the sums around 0 are the best candidates.
            # We can check a few neighbors.
            found = False
            for di in [0, 1, -1]:
                for dj in [0, 1, -1]:
                    ni = i + di
                    nj = j + dj
                    if 0 <= ni < n_a and 0 <= nj < n_b:
                        val = sums_a[ni] + sums_b[nj]
                        if val != 0:
                            if best_val is None or abs(val) < best_val:
                                best_val = abs(val)
                                best_a = sums_a[ni]
                                best_b = sums_b[nj]
                            if best_val == 1:
                                found = True
                                break
                if found:
                    break
            if found:
                break
            # If we found 0, we should move pointers to avoid staying at 0?
            # The next best sums are adjacent.
            i += 1
            j -= 1
        elif s > 0:
            # Sum is too large, decrease it by reducing b
            # But we should check this sum as a candidate first
            if s != 0:
                if best_val is None or s < best_val:
                    best_val = s
                    best_a = sums_a[i]
                    best_b = sums_b[j]
                    if best_val == 1:
                        break
            j -= 1
        else:
            # Sum is too small, increase it by increasing a
            if s != 0:
                if best_val is None or -s < best_val:
                    best_val = -s
                    best_a = sums_a[i]
                    best_b = sums_b[j]
                    if best_val == 1:
                        break
            i += 1
            
    # If we found a solution, reconstruct coefficients
    if best_a is not None and best_b is not None:
        coeffs_a = _find_coeffs(best_a, 0, mid, W)
        coeffs_b = _find_coeffs(best_b, mid, n, W)
        
        if coeffs_a is not None and coeffs_b is not None:
            final_coeffs = coeffs_a + coeffs_b
            total = sum(c * W[i] for i, c in enumerate(final_coeffs))
            if total != 0:
                return final_coeffs
    
    return _fallback(W, n)

def _find_coeffs(target, start, end, W):
    """Recursively find coefficients for indices start to end (0-based, exclusive end)."""
    if start == end:
        return [] if target == 0 else None
        
    # Pruning: max possible absolute sum from start to end
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
        # Make nonzero by setting smallest weight to 1
        min_idx = min(range(n), key=lambda i: W[i])
        coeffs[min_idx] = 1
        curr = sum(c * W[i] for i, c in enumerate(coeffs))
    
    return coeffs
