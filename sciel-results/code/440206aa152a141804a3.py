import time
from math import gcd
from functools import reduce
import itertools

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    # 1. Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    
    # Weights for first 15 terms (1..15)
    W_A = [L // i for i in range(1, 16)]
    # Weights for next 15 terms (16..30)
    W_B = [L // i for i in range(16, 31)]
    
    # 2. Enumerate all sums for A and store in a dictionary
    # Map sum_value -> base-3 index (0 to 3^15 - 1)
    sum_to_idx_A = {}
    
    # Generate all combinations for A
    # Using itertools.product is efficient for iteration
    # We encode the coefficients -1, 0, 1 as 0, 1, 2 for index calculation
    for coeffs_A in itertools.product(range(3), repeat=15):
        s = 0
        # Unroll loop for speed? 15 iterations is small.
        for i in range(15):
            c = coeffs_A[i] - 1
            if c != 0:
                s += c * W_A[i]
        
        if s not in sum_to_idx_A:
            # Encode index in base 3
            idx = 0
            for d in coeffs_A:
                idx = idx * 3 + d
            sum_to_idx_A[s] = idx
        
        if time.time() - start_time > deadline - 10:
            break # Safety break, though 3^15 is fast
    
    # 3. Enumerate sums for B and search for match
    # We want S_A + S_B = K for small non-zero K.
    # Check K = 1, -1, 2, -2, ...
    # Since we want minimum |K|, we check in order of |K|.
    
    best_K = None
    best_idx_a = None
    best_idx_b = None
    
    # Pre-calculate indices for B to avoid re-calculating if we need to check multiple K values?
    # No, it's better to iterate B sums once and check all target Ks?
    # Or iterate Ks and check B sums?
    # Iterating B sums (3^15 times) and doing a dict lookup is O(1).
    # Checking multiple Ks per B sum is just multiple dict lookups.
    
    # Let's generate B sums and for each, check if 1-S_B, -1-S_B, 2-S_B, -2-S_B... are in sum_to_idx_A
    # We can stop as soon as we find the smallest |K|.
    # But we don't know if a smaller |K| exists for a *later* B sum if we find K=2 for an *early* B sum.
    # So we must track the global minimum |K| found across all B sums.
    
    min_abs_K_found = float('inf')
    
    for coeffs_B in itertools.product(range(3), repeat=15):
        if time.time() - start_time > deadline - 5:
            break
            
        s_B = 0
        for i in range(15):
            c = coeffs_B[i] - 1
            if c != 0:
                s_B += c * W_B[i]
        
        # Check small K values
        # We only need to check K up to the current best min_abs_K_found
        # Start with K=1, -1
        # If we find K=1, we can return immediately as it's the global minimum possible non-zero.
        
        # Check K = 1
        target_A = 1 - s_B
        if target_A in sum_to_idx_A:
            # Found K=1, which is optimal
            idx_A = sum_to_idx_A[target_A]
            idx_B = 0
            for d in coeffs_B:
                idx_B = idx_B * 3 + d
            return _reconstruct(idx_A, idx_B)
        
        # Check K = -1
        target_A = -1 - s_B
        if target_A in sum_to_idx_A:
            # Found K=-1, which is optimal
            idx_A = sum_to_idx_A[target_A]
            idx_B = 0
            for d in coeffs_B:
                idx_B = idx_B * 3 + d
            return _reconstruct(idx_A, idx_B)
        
        # If 1 and -1 not found, check larger Ks if they are smaller than current best
        # However, 1 is almost certainly found. 
        # If not, we need to be more thorough.
        # Let's check up to K=10 for now, tracking the best.
        for k in range(2, 11):
            if k >= min_abs_K_found:
                break
                
            target_A_pos = k - s_B
            if target_A_pos in sum_to_idx_A:
                min_abs_K_found = k
                best_K = k
                best_idx_a = sum_to_idx_A[target_A_pos]
                idx_B = 0
                for d in coeffs_B:
                    idx_B = idx_B * 3 + d
                best_idx_b = idx_B
                
            target_A_neg = -k - s_B
            if target_A_neg in sum_to_idx_A:
                min_abs_K_found = k
                best_K = -k
                best_idx_a = sum_to_idx_A[target_A_neg]
                idx_B = 0
                for d in coeffs_B:
                    idx_B = idx_B * 3 + d
                best_idx_b = idx_B

    if best_K is not None:
        return _reconstruct(best_idx_a, best_idx_b)
    
    # Fallback if no small K found (very unlikely)
    # Return a valid non-zero sum, e.g., c_1=1, others 0 -> sum = 1
    # Or better, just return the seed's fallback which is valid.
    return [0] * 28 + [1, -1]

def _reconstruct(idx_A, idx_B):
    coeffs = [0] * 30
    # Decode idx_A (first 15 coefficients)
    idx = idx_A
    for i in range(14, -1, -1):
        d = idx % 3
        coeffs[i] = d - 1
        idx //= 3
        
    # Decode idx_B (last 15 coefficients)
    idx = idx_B
    for i in range(29, 14, -1):
        d = idx % 3
        coeffs[i] = d - 1
        idx //= 3
        
    return coeffs
