"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
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
        sums = [0]
        for w in half_weights:
            new_sums = []
            # Optimization: use list comprehension or extend
            # But explicit loop is fine for 3^15
            for s in sums:
                new_sums.append(s - w)
                new_sums.append(s)
                new_sums.append(s + w)
            sums = new_sums
        return sums

    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    sums1 = enumerate_sums(half1)
    sums1.sort()
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    # We don't need to sort sums2 if we iterate through it
    # But we need to generate it.
    # To save memory and time, we can generate sums2 on the fly?
    # No, generating 3^15 sums is fast. Let's just generate and iterate.
    sums2 = enumerate_sums(half2)
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    best_abs = None
    best_s1 = None
    best_s2 = None
    
    n1 = len(sums1)
    
    # Iterate through sums2
    for s2 in sums2:
        if time.time() > deadline:
            break
            
        # We want s1 such that s1 + s2 is close to 0
        # So s1 should be close to -s2
        target = -s2
        
        # Find insertion point for target in sums1
        pos = bisect_left(sums1, target)
        
        # Check neighbors: pos-1, pos, pos+1 (if valid)
        # We need to check the closest value to target
        candidates = []
        if pos > 0:
            candidates.append(pos - 1)
        if pos < n1:
            candidates.append(pos)
        if pos + 1 < n1:
            candidates.append(pos + 1)
            
        for idx in candidates:
            s1 = sums1[idx]
            total = s1 + s2
            if total != 0:
                abs_total = abs(total)
                if best_abs is None or abs_total < best_abs:
                    best_abs = abs_total
                    best_s1 = s1
                    best_s2 = s2
                    if best_abs == 1:
                        # Found the theoretical minimum
                        # We can break out of all loops
                        # But we are in a for loop, so we need to handle this
                        pass
    
    # Check if we found 1
    if best_abs == 1:
        pass # We can break early if we implement a flag, but let's just let it run or break
    
    # Actually, breaking early from the for loop is better
    # Let's restructure to allow early exit
    
    # Re-do search with early exit
    best_abs = None
    best_s1 = None
    best_s2 = None
    found_min = False
    
    for s2 in sums2:
        if time.time() > deadline:
            break
        if found_min:
            break
            
        target = -s2
        pos = bisect_left(sums1, target)
        
        candidates = []
        if pos > 0:
            candidates.append(pos - 1)
        if pos < n1:
            candidates.append(pos)
        if pos + 1 < n1:
            candidates.append(pos + 1)
            
        for idx in candidates:
            s1 = sums1[idx]
            total = s1 + s2
            if total != 0:
                abs_total = abs(total)
                if best_abs is None or abs_total < best_abs:
                    best_abs = abs_total
                    best_s1 = s1
                    best_s2 = s2
                    if best_abs == 1:
                        found_min = True
                        break
    
    if best_abs is None:
        return [0] * 28 + [1, -1]
    
    def decode_sum(half_weights, target_sum):
        n_h = len(half_weights)
        coeffs = [0] * n_h
        def backtrack(pos, remaining):
            if pos == n_h:
                return remaining == 0
            w = half_weights[pos]
            for c in (1, 0, -1):
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
    
    return result

# EVOLVE-BLOCK-END
