import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 30
    half = 15
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    # weights[i] = L // (i+1) for coefficient at position i (0-indexed), term 1/(i+1)
    weights = [L // (i + 1) for i in range(n)]
    
    w1 = weights[:half]   # first 15
    w2 = weights[half:]   # last 15
    
    def enumerate_sums(weights_half):
        """Enumerate all 3^len(weights_half) signed sums and their coefficient assignments.
        Returns (sums_list, coeffs_list) where coeffs_list[k] is a tuple of 15 values in {-1,0,1}."""
        sums = [0]
        coeffs = [()]
        for w in weights_half:
            new_sums = []
            new_coeffs = []
            for s, c in zip(sums, coeffs):
                # -1
                new_sums.append(s - w)
                new_coeffs.append(c + (-1,))
                # 0
                new_sums.append(s)
                new_coeffs.append(c + (0,))
                # +1
                new_sums.append(s + w)
                new_coeffs.append(c + (1,))
            sums = new_sums
            coeffs = new_coeffs
            if time.time() > deadline:
                break
        return sums, coeffs
    
    # Enumerate half 1
    sums1, coeffs1 = enumerate_sums(w1)
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    # Sort half 1 by sum value, keep track of original index for coefficient lookup
    order = sorted(range(len(sums1)), key=lambda i: sums1[i])
    sums1_sorted = [sums1[i] for i in order]
    coeffs1_sorted = [coeffs1[i] for i in order]
    
    # Enumerate half 2
    sums2, coeffs2 = enumerate_sums(w2)
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    import bisect
    
    best_abs = float('inf')
    best_total = 0
    best_idx1 = -1
    best_idx2 = -1
    
    for idx2, (s2, c2) in enumerate(zip(sums2, coeffs2)):
        target = -s2
        # Find position in sums1_sorted
        pos = bisect.bisect_left(sums1_sorted, target)
        
        # Check predecessor (pos-1) and successor (pos)
        for check_pos in [pos - 1, pos]:
            if 0 <= check_pos < len(sums1_sorted):
                s1_val = sums1_sorted[check_pos]
                total = s1_val + s2
                if total == 0:
                    continue
                abs_total = abs(total)
                if abs_total < best_abs:
                    best_abs = abs_total
                    best_total = total
                    best_idx1 = check_pos
                    best_idx2 = idx2
    
    if best_idx1 == -1:
        # Fallback: find any non-zero
        for idx2, (s2, c2) in enumerate(zip(sums2, coeffs2)):
            target = -s2
            pos = bisect.bisect_left(sums1_sorted, target)
            for check_pos in [pos - 1, pos]:
                if 0 <= check_pos < len(sums1_sorted):
                    s1_val = sums1_sorted[check_pos]
                    total = s1_val + s2
                    if total != 0:
                        best_abs = abs(total)
                        best_total = total
                        best_idx1 = check_pos
                        best_idx2 = idx2
                        break
            if best_idx1 != -1:
                break
    
    if best_idx1 == -1:
        return [0] * 28 + [1, -1]
    
    c1 = coeffs1_sorted[best_idx1]
    c2 = coeffs2_sorted[best_idx2]
    
    result = list(c1) + list(c2)
    
    # Verify
    total_check = 0
    for i, c in enumerate(result):
        if c != 0:
            total_check += c * weights[i]
    assert total_check == best_total, f"Mismatch: {total_check} vs {best_total}"
    assert total_check != 0
    
    return result
