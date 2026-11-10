import math
import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s limit
    
    n = 30
    
    # 1. Calculate LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = L * i // math.gcd(L, i)
    
    # Precompute weights: weight[i] = L // (i+1)
    weights = [L // (i + 1) for i in range(n)]
    
    # 2. Meet-in-the-Middle Enumeration
    # Split into two halves of 15
    mid = n // 2
    
    def generate_half_sums(start_idx, end_idx):
        """
        Generate all possible integer sums for coefficients at indices start_idx..end_idx-1.
        """
        num_items = end_idx - start_idx
        sums = [0]
        for j in range(num_items):
            w = weights[start_idx + j]
            new_sums = []
            for s in sums:
                new_sums.append(s - w)
                new_sums.append(s)
                new_sums.append(s + w)
            sums = new_sums
        return sums

    # Check time
    if time.time() > deadline - 10:
        return [0] * 28 + [1, -1]

    # Enumerate first half (indices 0..14)
    half1_sums = generate_half_sums(0, mid)
    
    # Enumerate second half (indices 15..29)
    half2_sums = generate_half_sums(mid, n)
    
    # 3. Find optimal pair
    # We want s1 + s2 != 0 minimizing |s1 + s2|
    # Sort half2_sums for binary search
    half2_sorted = sorted(half2_sums)
    
    best_delta = None
    best_s1 = None
    best_s2 = None
    
    # Iterate through all sums in half1
    for s1 in half1_sums:
        target = -s1
        # Find insertion point in half2_sorted
        pos = bisect.bisect_left(half2_sorted, target)
        
        # Check neighbors: pos-1, pos, pos+1
        # We need s2 such that s1 + s2 is close to 0 but not 0.
        candidates = []
        if pos > 0:
            candidates.append(pos - 1)
        if pos < len(half2_sorted):
            candidates.append(pos)
        if pos + 1 < len(half2_sorted):
            candidates.append(pos + 1)
        
        for p in candidates:
            s2 = half2_sorted[p]
            delta = s1 + s2
            if delta == 0:
                continue
            abs_delta = abs(delta)
            if best_delta is None or abs_delta < best_delta:
                best_delta = abs_delta
                best_s1 = s1
                best_s2 = s2
                # If we found delta=1, that's the absolute minimum possible (since L is integer)
                if best_delta == 1:
                    break
        if best_delta == 1:
            break
        
        if time.time() > deadline - 5:
            break
    
    if best_delta is None:
        # Fallback: should not happen
        return [0] * 28 + [1, -1]
    
    # 4. Recover Coefficients
    # We need to find coeffs c[0..mid-1] such that sum c[i]*weights[i] = best_s1
    # and coeffs c[mid..n-1] such that sum c[i]*weights[i] = best_s2
    
    def recover_coeffs(start_idx, end_idx, target_sum):
        """
        Recover coefficients that produce target_sum for the given index range.
        Uses meet-in-the-middle for the recovery step.
        """
        num_items = end_idx - start_idx
        mid_rec = num_items // 2
        
        # Generate sums for first part of recovery
        sums_r1 = [0]
        for j in range(mid_rec):
            w = weights[start_idx + j]
            new_sums = []
            for s in sums_r1:
                new_sums.append(s - w)
                new_sums.append(s)
                new_sums.append(s + w)
            sums_r1 = new_sums
        
        # Generate sums for second part of recovery
        sums_r2 = [0]
        for j in range(mid_rec, num_items):
            w = weights[start_idx + j]
            new_sums = []
            for s in sums_r2:
                new_sums.append(s - w)
                new_sums.append(s)
                new_sums.append(s + w)
            sums_r2 = new_sums
        
        # Find pair s_r1 + s_r2 = target_sum
        sums_r2_set = set(sums_r2)
        
        target_r1 = None
        target_r2 = None
        
        for s_r1 in sums_r1:
            needed_r2 = target_sum - s_r1
            if needed_r2 in sums_r2_set:
                target_r1 = s_r1
                target_r2 = needed_r2
                break
        
        if target_r1 is None:
            return None
        
        # Now recover coeffs for r1 and r2 using simple backtrack (small enough)
        def recover_small(start_idx, end_idx, target_sum):
            num_items = end_idx - start_idx
            coeffs = [0] * num_items
            
            def bt(idx, curr):
                if idx == num_items:
                    return curr == target_sum
                w = weights[start_idx + idx]
                coeffs[idx] = -1
                if bt(idx + 1, curr - w):
                    return True
                coeffs[idx] = 0
                if bt(idx + 1, curr):
                    return True
                coeffs[idx] = 1
                if bt(idx + 1, curr + w):
                    return True
                coeffs[idx] = 0
                return False
            
            if bt(0, 0):
                return coeffs
            return None
        
        c1 = recover_small(start_idx, start_idx + mid_rec, target_r1)
        c2 = recover_small(start_idx + mid_rec, end_idx, target_r2)
        
        if c1 is None or c2 is None:
            return None
        
        return c1 + c2
    
    half1_coeffs = recover_coeffs(0, mid, best_s1)
    half2_coeffs = recover_coeffs(mid, n, best_s2)
    
    if half1_coeffs is None or half2_coeffs is None:
        # Fallback
        return [0] * 28 + [1, -1]
    
    result = half1_coeffs + half2_coeffs
    
    # Verify
    total = sum(c * w for c, w in zip(result, weights))
    if total == 0:
        # Should not happen, but fix
        result[0] = 1
        total = sum(c * w for c, w in zip(result, weights))
    
    return result
