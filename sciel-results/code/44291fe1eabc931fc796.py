import time
import bisect
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    n = 30
    # Compute D = lcm(1, 2, ..., 30)
    def lcm(a, b):
        return a * b // gcd(a, b)
    D = 1
    for i in range(1, n + 1):
        D = lcm(D, i)
    
    # weights[i-1] = D // i, so sum of c[i-1]*weights[i-1] gives numerator
    weights = [D // i for i in range(1, n + 1)]
    
    # Split into two halves
    half = n // 2  # 15
    w1 = weights[:half]
    w2 = weights[half:]
    
    # Generate all sums for first half
    def gen_sums(ws):
        """Generate all possible sums using coefficients in {-1, 0, 1}"""
        sums = [0]
        for w in ws:
            new_sums = []
            for s in sums:
                new_sums.append(s)
                new_sums.append(s + w)
                new_sums.append(s - w)
            sums = new_sums
        return sums
    
    # Generate first half sums
    t0 = time.time()
    sums1 = gen_sums(w1)
    
    if time.time() > deadline:
        # Fallback: simple heuristic
        return [0] * 28 + [1, -1]
    
    # Generate second half sums and sort
    t1 = time.time()
    sums2 = gen_sums(w2)
    sums2.sort()
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    # For each sum in sums1, find best sum in sums2 to make total closest to 0 but nonzero
    best_abs = None
    best_s1 = 0
    best_s2 = 0
    
    for s1 in sums1:
        # We want s1 + s2 close to 0, so s2 close to -s1
        target = -s1
        # Binary search in sums2 for target
        idx = bisect.bisect_left(sums2, target)
        
        candidates = []
        if idx < len(sums2):
            candidates.append(sums2[idx])
        if idx > 0:
            candidates.append(sums2[idx - 1])
        # Also check idx+1 in case
        if idx + 1 < len(sums2):
            candidates.append(sums2[idx + 1])
        
        for s2 in candidates:
            total = s1 + s2
            if total == 0:
                continue
            abs_total = abs(total)
            if best_abs is None or abs_total < best_abs:
                best_abs = abs_total
                best_s1 = s1
                best_s2 = s2
    
    if best_abs is None:
        # Shouldn't happen, but fallback
        return [0] * 28 + [1, -1]
    
    # Now reconstruct the coefficients
    # Find coefficients for first half that give best_s1
    def find_coeffs(ws, target):
        """Find coefficients in {-1,0,1} that sum to target using weights ws"""
        n_w = len(ws)
        # Use DFS with pruning
        coeffs = [0] * n_w
        def dfs(i, current):
            if i == n_w:
                return current == target
            for c in (-1, 0, 1):
                coeffs[i] = c
                if dfs(i + 1, current + c * ws[i]):
                    return True
            coeffs[i] = 0
            return False
        if dfs(0, 0):
            return coeffs
        return None
    
    t2 = time.time()
    coeffs1 = find_coeffs(w1, best_s1)
    if coeffs1 is None or time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    coeffs2 = find_coeffs(w2, best_s2)
    if coeffs2 is None or time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    result = coeffs1 + coeffs2
    return result
