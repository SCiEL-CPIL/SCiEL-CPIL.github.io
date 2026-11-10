import numpy as np
from math import gcd
from functools import reduce
import time

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    return reduce(_lcm, lst, 1)

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin

    # 1. Calculate LCM of 1..30
    L = _lcm_list(range(1, 31))
    
    # 2. Calculate weights W_i = L / i
    # Indices 0..29 correspond to denominators 1..30
    weights = np.array([L // (i + 1) for i in range(30)], dtype=np.int64)
    
    # 3. Split into two halves
    n = 30
    n1 = n // 2
    n2 = n - n1
    
    w1 = weights[:n1]
    w2 = weights[n1:]
    
    def generate_sums_and_codes(weights_half):
        """
        Generates all possible sums and their base-3 encoded coefficient vectors.
        Returns:
        sums: np.array of int64
        codes: np.array of int32
        """
        # Start with sum=0, code=0
        # Code digit j represents coeff at position j in this half.
        # Mapping: -1 -> 0, 0 -> 1, 1 -> 2
        
        sums = np.array([0], dtype=np.int64)
        codes = np.array([0], dtype=np.int32)
        
        for j, w in enumerate(weights_half):
            power = 3 ** j
            # Branch 0: coeff = -1 (value 0 in base 3)
            # Branch 1: coeff = 0  (value 1 in base 3)
            # Branch 2: coeff = 1  (value 2 in base 3)
            
            # New sums
            s_keep = sums           # coeff 0 -> +1*power in code? No.
            s_add  = sums + w       # coeff 1 -> +2*power in code
            s_sub  = sums - w       # coeff -1 -> +0*power in code
            
            # Wait, let's re-verify the code mapping.
            # We want to reconstruct coeffs later.
            # If we store the code, we decode it.
            # Let's stick to:
            # Branch 'keep' (c=0): code += 1 * 3^j
            # Branch 'add'  (c=1): code += 2 * 3^j
            # Branch 'sub'  (c=-1): code += 0 * 3^j
            
            c_keep = codes + 1 * power
            c_add  = codes + 2 * power
            c_sub  = codes + 0 * power
            
            sums = np.concatenate([s_keep, s_add, s_sub])
            codes = np.concatenate([c_keep, c_add, c_sub])
            
            # Optional: prune if time is running out? 
            # No, 3^15 is small enough.
            
        return sums, codes

    # Generate for first half
    sums1, codes1 = generate_sums_and_codes(w1)
    
    # Check time
    if time.time() - start_time > DEADLINE - 30:
         # Fallback to simple local search if generation is unexpectedly slow
         pass 

    # Generate for second half
    sums2, codes2 = generate_sums_and_codes(w2)
    
    # Sort by sums
    order1 = np.argsort(sums1)
    sums1 = sums1[order1]
    codes1 = codes1[order1]
    
    order2 = np.argsort(sums2)
    sums2 = sums2[order2]
    codes2 = codes2[order2]
    
    # Search for best pair
    # We want min |s1 + s2| != 0
    best_val = None
    best_code1 = None
    best_code2 = None
    
    # Use bisect for speed
    import bisect
    
    # sums2 is a numpy array, bisect works on it but might be slower than list?
    # Converting to list is fast for 14M elements? ~0.5s
    # Let's stick to numpy searchsorted which is optimized.
    
    for i in range(len(sums1)):
        s1 = sums1[i]
        target = -s1
        
        # Find insertion point in sums2
        pos = np.searchsorted(sums2, target)
        
        # Check pos and pos-1
        candidates = []
        if pos < len(sums2):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
            
        # Also check neighbors of pos in case of duplicates or close values
        # Since we want closest to target, pos and pos-1 are the primary candidates.
        # However, if sums2 has duplicates, or if target is between values,
        # checking pos and pos-1 is sufficient for "closest".
        
        for p in candidates:
            s2 = sums2[p]
            total = s1 + s2
            
            if total == 0:
                continue
                
            abs_total = abs(total)
            
            if best_val is None or abs_total < best_val:
                best_val = abs_total
                best_code1 = codes1[i]
                best_code2 = codes2[p]
                
    # Decode coefficients
    coeffs = [0] * 30
    
    # Decode code1 (first n1 coeffs)
    c1 = best_code1
    for j in range(n1):
        digit = c1 % 3
        c1 //= 3
        # Map back: 0 -> -1, 1 -> 0, 2 -> 1
        if digit == 0:
            coeffs[j] = -1
        elif digit == 1:
            coeffs[j] = 0
        else:
            coeffs[j] = 1
            
    # Decode code2 (next n2 coeffs)
    c2 = best_code2
    for j in range(n2):
        digit = c2 % 3
        c2 //= 3
        if digit == 0:
            coeffs[n1 + j] = -1
        elif digit == 1:
            coeffs[n1 + j] = 0
        else:
            coeffs[n1 + j] = 1
            
    return coeffs
