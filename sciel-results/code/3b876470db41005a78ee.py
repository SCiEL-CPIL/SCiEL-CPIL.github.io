import math
import bisect
import time

def construct():
    start_time = time.time()
    limit = 50.0 # Safety margin for 60s limit
    
    n = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = L * i // math.gcd(L, i)
        
    # Weights
    W = [L // i for i in range(1, n + 1)]
    
    # Split into two halves
    half = n // 2
    W1 = W[:half]
    W2 = W[half:]
    
    # Generate sums for first half
    # We need to store (sum, code)
    # code is an integer encoding the coefficients in base 3
    # coeff = -1 -> 0, 0 -> 1, 1 -> 2
    
    def generate_sums(weights):
        sums = []
        # Iterative generation to avoid recursion depth issues and overhead
        # Start with sum=0, code=0
        # But we need to expand.
        # Let's use a list of (sum, code) and expand it level by level.
        
        current = [(0, 0)]
        for i, w in enumerate(weights):
            next_level = []
            for s, c in current:
                # -w, c*3 + 0
                next_level.append((s - w, c * 3))
                # 0, c*3 + 1
                next_level.append((s, c * 3 + 1))
                # +w, c*3 + 2
                next_level.append((s + w, c * 3 + 2))
            current = next_level
        return current

    sums1 = generate_sums(W1)
    sums2 = generate_sums(W2)
    
    # Sort sums2 by sum value
    sums2.sort(key=lambda x: x[0])
    
    # Extract just the sums for binary search
    sums2_vals = [s[0] for s in sums2]
    
    best_dist = float('inf')
    best_code1 = None
    best_code2 = None
    
    for s1, c1 in sums1:
        target = -s1
        # Binary search for target in sums2_vals
        pos = bisect.bisect_left(sums2_vals, target)
        
        # Check pos and pos-1
        candidates = []
        if pos < len(sums2_vals):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
            
        for p in candidates:
            s2_val = sums2_vals[p]
            dist = abs(s1 + s2_val)
            if dist == 0:
                # We need non-zero sum. 
                # If we find 0, we must look for the next closest non-zero.
                # However, the probability of hitting exactly 0 is low but possible.
                # If dist == 0, we cannot use this pair.
                # We should check neighbors further away if dist == 0.
                # But usually, if 0 is found, the next best is likely nearby.
                # Let's just skip 0 for now and see if we find a non-zero.
                # Actually, if we skip 0, we might miss a very small non-zero sum 
                # that is not an immediate neighbor if 0 is present?
                # No, if 0 is present, the sorted list has ... 0 ... 
                # The neighbors of 0 are the smallest non-zero values.
                # So checking pos and pos-1 is sufficient to find the closest to 0.
                # If one of them is 0, the other is the closest non-zero?
                # Not necessarily. If the list is [0, 1], target 0.
                # pos=0. candidates: 0 (val 0, dist 0), -1 (out of bounds).
                # We need to handle the case where the closest is 0.
                # In that case, we need the next closest.
                # Let's check pos+1 and pos-2 as well if dist==0?
                # Better: collect all candidates in range [pos-2, pos+2] and filter dist > 0.
                pass
            
            if dist > 0 and dist < best_dist:
                best_dist = dist
                best_code1 = c1
                best_code2 = sums2[p][1]
                
    # If best_dist is still inf, it means we only found 0 sums? Unlikely.
    # Let's refine the candidate check to handle dist=0 properly.
    
    # Re-run search logic with better candidate handling
    best_dist = float('inf')
    best_code1 = None
    best_code2 = None
    
    for s1, c1 in sums1:
        target = -s1
        pos = bisect.bisect_left(sums2_vals, target)
        
        # Check a small window around pos
        # Since we want the closest non-zero, and 0 might be at pos,
        # we check pos-1, pos, pos+1.
        # If any of these is 0, we ignore it.
        # The next closest will be further out, but given the density,
        # checking a small window is usually enough.
        # To be safe, check pos-2 to pos+2.
        
        start = max(0, pos - 2)
        end = min(len(sums2_vals), pos + 3)
        
        for p in range(start, end):
            s2_val = sums2_vals[p]
            dist = abs(s1 + s2_val)
            if dist > 0 and dist < best_dist:
                best_dist = dist
                best_code1 = c1
                best_code2 = sums2[p][1]
                
    if best_code1 is None:
        # Fallback: return seed
        return [0] * 28 + [1, -1]
        
    # Decode codes to coefficients
    def decode_code(code, length):
        coeffs = [0] * length
        for i in range(length):
            val = code % 3
            code //= 3
            # 0 -> -1, 1 -> 0, 2 -> 1
            if val == 0:
                coeffs[i] = -1
            elif val == 1:
                coeffs[i] = 0
            else:
                coeffs[i] = 1
        return coeffs
        
    c1_coeffs = decode_code(best_code1, half)
    c2_coeffs = decode_code(best_code2, n - half)
    
    return c1_coeffs + c2_coeffs
