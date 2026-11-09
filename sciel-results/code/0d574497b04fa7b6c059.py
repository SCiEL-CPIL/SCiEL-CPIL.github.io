"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    n = 30
    # Calculate LCM of 1..30
    lcm_val = 1
    for i in range(1, n + 1):
        lcm_val = lcm_val * i // gcd(lcm_val, i)
        
    # Weights for integer arithmetic: S = sum(c_i / i) = (1/LCM) * sum(c_i * LCM/i)
    # c_i in {-1, 0, 1}
    # weights[i-1] corresponds to denominator i
    weights = [lcm_val // i for i in range(1, n + 1)]
    
    # Split into two halves
    half = 15
    w1_indices = list(range(0, half))
    w2_indices = list(range(half, n))
    
    w1 = [weights[i] for i in w1_indices]
    w2 = [weights[i] for i in w2_indices]
    
    def gen_sums(ws):
        sums = [0]
        for w in ws:
            # Optimization: Use list comprehension or extend
            # sums = [s - w for s in sums] + sums + [s + w for s in sums]
            # This creates 3 lists. 
            new_sums = []
            # Pre-allocate size? No, list append is amortized O(1)
            for s in sums:
                new_sums.append(s - w)
                new_sums.append(s)
                new_sums.append(s + w)
            sums = new_sums
        return sums

    t0 = time.time()
    sums1 = gen_sums(w1)
    sums1.sort()
    t1 = time.time()
    
    if time.time() > deadline - 10:
        # Fallback if generation is too slow
        # Return a valid nonzero sum, e.g., c_1=1, others 0 -> S=1
        # Coeffs: [1, 0, ..., 0]
        return [1] + [0]*(n-1)
        
    sums2 = gen_sums(w2)
    sums2.sort()
    
    t2 = time.time()
    
    # Two-pointer scan to find min |a+b| != 0
    i = 0
    j = len(sums2) - 1
    n1 = len(sums1)
    n2 = len(sums2)
    
    best_abs = float('inf')
    best_a = None
    best_b = None
    
    while i < n1 and j >= 0:
        val = sums1[i] + sums2[j]
        if val == 0:
            # Check neighbors to find smallest non-zero
            # Left neighbor in S2 (smaller b)
            if j > 0:
                v = sums1[i] + sums2[j-1]
                if v != 0:
                    av = abs(v)
                    if av < best_abs:
                        best_abs = av
                        best_a = sums1[i]
                        best_b = sums2[j-1]
                        if best_abs == 1:
                            break
            # Right neighbor in S1 (larger a)
            if i < n1 - 1:
                v = sums1[i+1] + sums2[j]
                if v != 0:
                    av = abs(v)
                    if av < best_abs:
                        best_abs = av
                        best_a = sums1[i+1]
                        best_b = sums2[j]
                        if best_abs == 1:
                            break
            # Move pointers
            # If val == 0, S1[i] = -S2[j]
            # S1[i+1] >= S1[i] => S1[i+1] + S2[j] >= 0
            # S2[j-1] <= S2[j] => S1[i] + S2[j-1] <= 0
            # We checked the immediate neighbors.
            # Moving both pointers is generally safe for finding the global min in sorted arrays
            # because any other combination involving S1[i] or S2[j] will be further from 0
            # or we will encounter it in subsequent steps.
            # However, strictly speaking, S1[i+1] + S2[j-1] might be closer?
            # S1[i+1] >= S1[i], S2[j-1] <= S2[j].
            # Let S1[i] = -S2[j] = K.
            # S1[i+1] + S2[j-1] = (K + d1) + (-K - d2) = d1 - d2.
            # S1[i+1] + S2[j] = (K + d1) + (-K) = d1.
            # S1[i] + S2[j-1] = K + (-K - d2) = -d2.
            # We checked d1 and -d2.
            # d1 - d2 could be smaller than min(d1, d2)?
            # Example: d1=10, d2=1. |10|=10, |-1|=1. |10-1|=9. No.
            # Example: d1=5, d2=5. |5|=5, |-5|=5. |0|=0. (But 0 is excluded).
            # If d1=1, d2=2. |1|=1, |-2|=2. |-1|=1.
            # It seems checking immediate neighbors is sufficient for the "closest to 0" pair in 1D sorted arrays.
            i += 1
            j -= 1
        elif val > 0:
            if val < best_abs:
                best_abs = val
                best_a = sums1[i]
                best_b = sums2[j]
                if best_abs == 1:
                    break
            j -= 1
        else: # val < 0
            av = -val
            if av < best_abs:
                best_abs = av
                best_a = sums1[i]
                best_b = sums2[j]
                if best_abs == 1:
                    break
            i += 1
            
    # Reconstruct coefficients
    def find_coeffs(ws, indices, target):
        # ws: list of weights
        # indices: original indices in the full 30-coeff array
        # target: sum to achieve
        n = len(ws)
        # Sort weights in descending order for better pruning
        # Create list of (weight, original_index)
        pairs = [(ws[i], indices[i]) for i in range(n)]
        pairs.sort(key=lambda x: -x[0])
        
        coeffs = [0] * n # coeffs[i] corresponds to pairs[i]
        
        # Suffix sums of weights (sorted)
        suffix_sums = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + pairs[i][0]
            
        def dfs(idx, rem):
            if idx == n:
                return rem == 0
            if abs(rem) > suffix_sums[idx]:
                return False
            
            # Try 0, 1, -1. 
            # Heuristic: try 0 first if rem is close to 0? 
            # Or try the value that reduces |rem| the most?
            # rem - c * w.
            # If rem > 0, c=1 reduces rem. c=-1 increases. c=0 keeps.
            # If rem < 0, c=-1 reduces |rem|. c=1 increases.
            
            w = pairs[idx][0]
            
            # Order of checks:
            # 1. The sign that moves rem towards 0
            # 2. 0
            # 3. The sign that moves away
            
            if rem > 0:
                order = [1, 0, -1]
            elif rem < 0:
                order = [-1, 0, 1]
            else:
                order = [0, 1, -1]
                
            for c in order:
                coeffs[idx] = c
                if dfs(idx + 1, rem - c * w):
                    return True
            return False
            
        if dfs(0, target):
            # Map back to original indices
            res = [0] * 30
            for i in range(n):
                res[pairs[i][1]] = coeffs[i]
            return res
        return None
        
    if best_a is not None and best_b is not None:
        c1 = find_coeffs(w1, w1_indices, best_a)
        if c1 is None:
             # Fallback if reconstruction fails (shouldn't happen)
             return [1] + [0]*(n-1)
        c2 = find_coeffs(w2, w2_indices, best_b)
        if c2 is None:
             return [1] + [0]*(n-1)
        
        # Combine
        final_coeffs = [0] * 30
        for i in range(30):
            final_coeffs[i] = c1[i] + c2[i]
            
        # Verify
        total = 0
        for i in range(30):
            if final_coeffs[i] != 0:
                total += final_coeffs[i] * (lcm_val // (i+1))
                
        if total != 0:
            return final_coeffs
            
    # Fallback
    return [1] + [0]*(n-1)

# EVOLVE-BLOCK-END
