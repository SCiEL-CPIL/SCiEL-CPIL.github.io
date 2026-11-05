import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # 85 seconds to be safe against 90s limit

    n = 30
    half = 15

    # Precompute exact fractions for each term
    terms = [Fraction(1, i) for i in range(1, n + 1)]

    # Split into two halves: first 15 and last 15
    # Left: indices 0..14 (terms 1/1 .. 1/15)
    # Right: indices 15..29 (terms 1/16 .. 1/30)

    # Generate all 3^15 sums for the left half
    left_sums = []
    for mask in range(3 ** half):
        s = Fraction(0)
        m = mask
        for i in range(half):
            digit = m % 3
            m //= 3
            if digit == 1:
                s += terms[i]
            elif digit == 2:
                s -= terms[i]
        left_sums.append(s)

    # Generate all 3^15 sums for the right half
    right_sums = []
    for mask in range(3 ** half):
        s = Fraction(0)
        m = mask
        for i in range(half):
            digit = m % 3
            m //= 3
            if digit == 1:
                s += terms[half + i]
            elif digit == 2:
                s -= terms[half + i]
        right_sums.append(s)

    # Sort right sums by absolute value to find the closest match to each left sum
    # We want to minimize |L + R| where L is from left_sums and R is from right_sums
    # Note: right_sums contains both positive and negative values.
    # Actually, it's easier to sort right_sums and for each L, binary search for -L in right_sums.
    
    right_sums.sort()
    
    best_abs = Fraction(1) # Initial guess, will be updated
    best_left_idx = 0
    best_right_idx = 0
    best_L = None
    best_R = None
    
    import bisect
    
    # For each left sum, find the right sum that makes the total closest to 0
    for L in left_sums:
        if (time.time() - start_time) >= deadline:
            break
            
        # We want R such that L + R is close to 0 => R is close to -L
        target = -L
        # Find insertion point in right_sums
        pos = bisect.bisect_left(right_sums, target)
        
        # Check pos and pos-1
        candidates = []
        if pos < len(right_sums):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
            
        for idx in candidates:
            R = right_sums[idx]
            total = L + R
            if total == 0:
                # Found zero sum, which is invalid. We need the next best non-zero.
                # But MITM finds the global minimum. If 0 is possible, we must exclude it.
                # The problem states "signed sum ... must be nonzero".
                # So we skip 0 and look for the smallest non-zero.
                continue
                
            abs_total = abs(total)
            if abs_total < best_abs:
                best_abs = abs_total
                best_left_idx = left_sums.index(L) # This is O(N), inefficient but N=143489. 
                # Better to track index during iteration? 
                # Let's refactor to track indices properly.
                
    # Refactor: The above loop has an O(N^2) risk due to .index(). Let's do it cleaner.
    best_abs = Fraction(1)
    best_L_val = None
    best_R_val = None
    best_L_mask = 0
    best_R_mask = 0
    
    # Regenerate with masks tracked? Or just store (sum, mask).
    # 3^15 is ~143k. Storing tuples is fine.
    
    left_data = []
    for mask in range(3 ** half):
        s = Fraction(0)
        m = mask
        for i in range(half):
            digit = m % 3
            m //= 3
            if digit == 1:
                s += terms[i]
            elif digit == 2:
                s -= terms[i]
        left_data.append((s, mask))
        
    right_data = []
    for mask in range(3 ** half):
        s = Fraction(0)
        m = mask
        for i in range(half):
            digit = m % 3
            m //= 3
            if digit == 1:
                s += terms[half + i]
            elif digit == 2:
                s -= terms[half + i]
        right_data.append((s, mask))
        
    # Sort right_data by sum
    right_data.sort(key=lambda x: x[0])
    right_sums_sorted = [x[0] for x in right_data]
    
    for L_val, L_mask in left_data:
        if (time.time() - start_time) >= deadline:
            break
            
        target = -L_val
        pos = bisect.bisect_left(right_sums_sorted, target)
        
        # Check neighbors
        for idx in [pos, pos - 1]:
            if 0 <= idx < len(right_data):
                R_val, R_mask = right_data[idx]
                total = L_val + R_val
                if total == 0:
                    continue
                abs_total = abs(total)
                if abs_total < best_abs:
                    best_abs = abs_total
                    best_L_val = L_val
                    best_R_val = R_val
                    best_L_mask = L_mask
                    best_R_mask = R_mask
    
    # Reconstruct coefficients from masks
    coeff = [0] * n
    
    # Decode left mask
    m = best_L_mask
    for i in range(half):
        digit = m % 3
        m //= 3
        if digit == 1:
            coeff[i] = 1
        elif digit == 2:
            coeff[i] = -1
            
    # Decode right mask
    m = best_R_mask
    for i in range(half):
        digit = m % 3
        m //= 3
        if digit == 1:
            coeff[half + i] = 1
        elif digit == 2:
            coeff[half + i] = -1
            
    # Verify result
    total = sum(coeff[i] * terms[i] for i in range(n))
    if total == 0:
        # Fallback if something went wrong (should not happen with MITM logic above)
        # Try a simple non-zero configuration
        coeff = [0] * 28 + [1, -1]
        total = sum(coeff[i] * terms[i] for i in range(n))
        
    return coeff
# EVOLVE-BLOCK-END
