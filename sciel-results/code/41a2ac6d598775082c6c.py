import time
from math import gcd
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, 31):
        L = L * i // gcd(L, i)
    
    A = [0] * (N + 1)
    for i in range(1, N + 1):
        A[i] = L // i
        
    # Split into two halves
    half1 = list(range(1, 16))
    half2 = list(range(16, 31))
    
    # Generate all sums for first half
    def gen_sums_list(indices):
        current_sums = [0]
        for i in indices:
            a_i = A[i]
            # Using list comprehension for speed
            add = [s + a_i for s in current_sums]
            sub = [s - a_i for s in current_sums]
            # Keep zeros by extending
            current_sums = add + sub + current_sums
        return current_sums

    sums1 = gen_sums_list(half1)
    sums1.sort()
    
    # Generate all sums for second half
    sums2 = gen_sums_list(half2)
    
    best_abs = float('inf')
    best_s1 = None
    best_s2 = None
    
    # Iterate through sums2 and find best match in sums1
    # We can optimize this by checking if we can get 0, but we must exclude 0.
    # So we look for the smallest non-zero absolute sum.
    
    for s2 in sums2:
        if time.time() > deadline - 5:
            break
        target = -s2
        pos = bisect.bisect_left(sums1, target)
        
        candidates = []
        if pos < len(sums1):
            candidates.append(sums1[pos])
        if pos > 0:
            candidates.append(sums1[pos-1])
            
        for s1 in candidates:
            total = s1 + s2
            if total != 0:
                abs_total = abs(total)
                if abs_total < best_abs:
                    best_abs = abs_total
                    best_s1 = s1
                    best_s2 = s2
                    if best_abs == 1:
                        return reconstruct(best_s1, best_s2, half1, half2, A)

    if best_s1 is None:
        return [0]*28 + [1, -1]
        
    return reconstruct(best_s1, best_s2, half1, half2, A)

def reconstruct(s1, s2, half1, half2, A):
    def find_coeffs(target, indices):
        n = len(indices)
        coeffs = [0] * n
        
        def dfs(idx, current_sum):
            if idx == n:
                return current_sum == target
            
            a_i = A[indices[idx]]
            
            # Pruning: check if it's possible to reach target with remaining items
            # Max possible sum from remaining items
            rem_max = sum(A[indices[j]] for j in range(idx+1, n))
            rem_min = -rem_max
            
            # If current_sum + rem_max < target or current_sum + rem_min > target, prune
            # Actually, we need to check if target - current_sum is within [rem_min, rem_max]
            # But since coefficients are -1, 0, 1, the reachable sums are not continuous.
            # However, the bound check is still valid for pruning.
            
            needed = target - current_sum
            if needed > rem_max or needed < rem_min:
                return False
            
            # Heuristic: try 0 first? Or 1/-1?
            # Let's try 1, -1, 0 or 0, 1, -1.
            # Usually, larger coefficients change the sum more, so trying them first might help find a solution faster if it exists.
            # But 0 is "safe" in some sense? No, it's just one option.
            
            # Try 1
            coeffs[idx] = 1
            if dfs(idx + 1, current_sum + a_i):
                return True
            
            # Try -1
            coeffs[idx] = -1
            if dfs(idx + 1, current_sum - a_i):
                return True
            
            # Try 0
            coeffs[idx] = 0
            if dfs(idx + 1, current_sum):
                return True
            
            return False
            
        if dfs(0, 0):
            return coeffs
        return None

    c1 = find_coeffs(s1, half1)
    c2 = find_coeffs(s2, half2)
    
    if c1 is None or c2 is None:
        return [0]*28 + [1, -1]
        
    result = [0] * 30
    for i, idx in enumerate(half1):
        result[idx-1] = c1[i]
    for i, idx in enumerate(half2):
        result[idx-1] = c2[i]
        
    return result
