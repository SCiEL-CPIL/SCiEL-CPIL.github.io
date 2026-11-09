import time
import numpy as np
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    n = 30
    # Calculate LCM of 1..30
    lcm_val = 1
    for i in range(1, n + 1):
        lcm_val = lcm_val * i // gcd(lcm_val, i)
        
    # Weights for integer arithmetic
    weights = [lcm_val // i for i in range(1, n + 1)]
    
    half = 15
    w1 = weights[:half]
    w2 = weights[half:]
    
    # Generate sums using NumPy for speed
    def gen_sums_np(ws):
        sums = np.array([0], dtype=np.int64)
        for w in ws:
            # Create three copies: -w, 0, +w added to existing sums
            # np.concatenate is fast
            sums = np.concatenate((sums - w, sums, sums + w))
        return sums

    t0 = time.time()
    sums1 = gen_sums_np(w1)
    sums2 = gen_sums_np(w2)
    
    # Sort sums2 to potentially speed up isin or just for fallback
    # numpy.isin uses a hash table or sorted search internally, so sorting might not be strictly necessary 
    # but it can help if we do manual binary search later.
    # Let's sort both for the fallback two-pointer method just in case.
    sums1.sort()
    sums2.sort()
    
    t1 = time.time()
    
    # Strategy 1: Check for small integer sums k = 1, 2, ...
    # If sum is k, then s1 + s2 = k => s2 = k - s1.
    # We check if (k - sums1) intersects with sums2.
    
    best_k = None
    best_s1 = None
    best_s2 = None
    
    # Limit k search to avoid infinite loop if 1 doesn't exist (unlikely)
    # 1000 is a safe upper bound for "small"
    for k in range(1, 1001):
        if time.time() > deadline - 10:
            break
            
        targets = k - sums1
        # Check intersection
        # numpy.isin is O(N log M) or similar
        matches = np.isin(targets, sums2)
        
        if np.any(matches):
            idx = np.argmax(matches)
            best_k = k
            best_s1 = sums1[idx]
            best_s2 = k - best_s1
            break
            
    # If we found a small k, reconstruct coefficients
    if best_k is not None:
        def find_coeffs(weights, target):
            n = len(weights)
            coeffs = [0] * n
            
            # Suffix sums for pruning
            suffix_sums = [0] * (n + 1)
            for i in range(n - 1, -1, -1):
                suffix_sums[i] = suffix_sums[i + 1] + weights[i]
                
            def dfs(idx, rem):
                if idx == n:
                    return rem == 0
                if abs(rem) > suffix_sums[idx]:
                    return False
                # Try -1, 0, 1
                # Heuristic: try 0 first? Or -1? 
                # Doesn't matter much for existence, but 0 might lead to dead ends less often?
                for c in [0, -1, 1]:
                    coeffs[idx] = c
                    if dfs(idx + 1, rem - c * weights[idx]):
                        return True
                return False
                
            if dfs(0, target):
                return coeffs
            return None
            
        c1 = find_coeffs(w1, best_s1)
        c2 = find_coeffs(w2, best_s2)
        
        if c1 and c2:
            return c1 + c2
            
    # Strategy 2: Fallback - Find the minimum absolute non-zero sum using two-pointer on sorted arrays
    # This guarantees finding the global minimum if Strategy 1 failed or timed out.
    i = 0
    j = len(sums2) - 1
    n1 = len(sums1)
    n2 = len(sums2)
    
    best_abs = float('inf')
    best_a = None
    best_b = None
    
    # Use a step to avoid iterating 14M times in Python if possible?
    # Two pointer is O(N). 14M steps in Python is ~1-2s. Acceptable.
    while i < n1 and j >= 0:
        if time.time() > deadline - 2:
            break
            
        val = sums1[i] + sums2[j]
        if val == 0:
            # Check neighbors for non-zero
            if j > 0:
                v = sums1[i] + sums2[j-1]
                if v != 0:
                    av = abs(v)
                    if av < best_abs:
                        best_abs = av
                        best_a = sums1[i]
                        best_b = sums2[j-1]
                        if best_abs == 1: break
            if i < n1 - 1:
                v = sums1[i+1] + sums2[j]
                if v != 0:
                    av = abs(v)
                    if av < best_abs:
                        best_abs = av
                        best_a = sums1[i+1]
                        best_b = sums2[j]
                        if best_abs == 1: break
            i += 1
            j -= 1
        elif val > 0:
            if val < best_abs:
                best_abs = val
                best_a = sums1[i]
                best_b = sums2[j]
                if best_abs == 1: break
            j -= 1
        else:
            av = -val
            if av < best_abs:
                best_abs = av
                best_a = sums1[i]
                best_b = sums2[j]
                if best_abs == 1: break
            i += 1
            
    if best_a is not None and best_b is not None:
        def find_coeffs(weights, target):
            n = len(weights)
            coeffs = [0] * n
            
            suffix_sums = [0] * (n + 1)
            for i in range(n - 1, -1, -1):
                suffix_sums[i] = suffix_sums[i + 1] + weights[i]
                
            def dfs(idx, rem):
                if idx == n:
                    return rem == 0
                if abs(rem) > suffix_sums[idx]:
                    return False
                for c in [0, -1, 1]:
                    coeffs[idx] = c
                    if dfs(idx + 1, rem - c * weights[idx]):
                        return True
                return False
                
            if dfs(0, target):
                return coeffs
            return None
            
        c1 = find_coeffs(w1, best_a)
        c2 = find_coeffs(w2, best_b)
        if c1 and c2:
            return c1 + c2

    # Final Fallback
    return [0] * (n - 2) + [1, -1]
