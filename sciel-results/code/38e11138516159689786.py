"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    n = 30
    # Compute L = lcm(1, 2, ..., 30)
    def lcm(a, b):
        return a // gcd(a, b) * b
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # Precompute L // i for each i
    Li = [L // i for i in range(1, n + 1)]
    
    # Split into two halves: first 15, last 15
    half = n // 2  # 15
    
    # Group A: indices 0..14 (terms 1..15)
    # Group B: indices 15..29 (terms 16..30)
    
    def enumerate_group(start_idx, end_idx):
        """Enumerate all 3^(end_idx-start_idx) combinations for the given range.
        Returns list of (sum_value, coeff_list) where coeff_list has length n with -1,0,1."""
        length = end_idx - start_idx
        results = []
        # Use iterative approach to avoid recursion depth issues
        # We'll use a base-3 counter
        count = 3 ** length
        for code in range(count):
            # Decode base-3
            temp = code
            coeffs = [0] * length
            for j in range(length):
                temp, rem = divmod(temp, 3)
                # rem is 0,1,2 -> map to -1,0,1
                coeffs[j] = rem - 1
            # Compute sum
            s = 0
            for j in range(length):
                if coeffs[j] != 0:
                    s += coeffs[j] * Li[start_idx + j]
            results.append((s, coeffs))
        return results
    
    # Check time
    if time.time() - start_time > DEADLINE:
        return [0] * 28 + [1, -1]
    
    # Enumerate group A (first 15 terms)
    groupA = enumerate_group(0, half)
    
    if time.time() - start_time > DEADLINE:
        return [0] * 28 + [1, -1]
    
    # Enumerate group B (last 15 terms)
    groupB = enumerate_group(half, n)
    
    if time.time() - start_time > DEADLINE:
        return [0] * 28 + [1, -1]
    
    # Sort group B by sum value
    groupB_sorted = sorted(groupB, key=lambda x: x[0])
    
    # For each entry in group A, find the entry in group B that minimizes |sumA + sumB|
    # where sumA + sumB != 0
    best_val = None
    best_coeffs = None
    
    # Use binary search on groupB_sorted
    import bisect
    
    B_values = [x[0] for x in groupB_sorted]
    
    for sumA, coeffsA in groupA:
        # We want sumA + sumB close to 0 but not 0
        # So sumB close to -sumA
        target = -sumA
        # Find insertion point
        pos = bisect.bisect_left(B_values, target)
        
        # Check positions around pos
        candidates = []
        for p in [pos - 1, pos, pos + 1]:
            if 0 <= p < len(B_values):
                candidates.append(p)
        
        for p in candidates:
            sumB, coeffsB = groupB_sorted[p]
            total = sumA + sumB
            if total == 0:
                continue
            if best_val is None or abs(total) < abs(best_val):
                best_val = total
                # Combine coefficients
                full_coeffs = [0] * n
                for j in range(half):
                    full_coeffs[j] = coeffsA[j]
                for j in range(half):
                    full_coeffs[half + j] = coeffsB[j]
                best_coeffs = full[:] if False else list(full_coeffs)
        
        if time.time() - start_time > DEADLINE:
            break
    
    if best_coeffs is not None:
        return best_coeffs
    else:
        return [0] * 28 + [1, -1]
# EVOLVE-BLOCK-END

