import math
import time
import bisect
from fractions import Fraction

def construct():
    """
    Returns exactly 30 integer coefficients from {-1, 0, 1} such that
    sum(coeff[i-1]/i for i in 1..30) is nonzero and has minimal absolute value.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 30
    # Split into two halves: indices 0..14 (denoms 1..15) and 15..29 (denoms 16..30)
    half1 = 15
    half2 = 15
    
    # Calculate LCM of 1 to 30
    def lcm(a, b):
        return a // math.gcd(a, b) * b
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # Scaled weights: weight[i] = L // (i+1)
    # Position i corresponds to denominator i+1
    weights = [L // (i + 1) for i in range(n)]
    
    def enumerate_half_dict(start, length):
        """
        Generates a dictionary mapping sum -> coeffs tuple for positions start..start+length-1.
        Uses iterative expansion. Only stores the first assignment found for each sum.
        """
        d = {0: ()}
        for pos in range(start, start + length):
            w = weights[pos]
            new_d = {}
            for s, c in d.items():
                # coeff = 0
                key0 = s
                if key0 not in new_d:
                    new_d[key0] = c + (0,)
                # coeff = 1
                key1 = s + w
                if key1 not in new_d:
                    new_d[key1] = c + (1,)
                # coeff = -1
                keym1 = s - w
                if keym1 not in new_d:
                    new_d[keym1] = c + (-1,)
            d = new_d
        return d

    # Enumerate first half
    d1 = enumerate_half_dict(0, half1)
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
        
    # Enumerate second half
    d2 = enumerate_half_dict(half1, half2)
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    # Sort keys of d2 for binary search
    keys2 = sorted(d2.keys())
    
    best_abs = None
    best_c1 = None
    best_c2 = None
    
    # Iterate over d1
    for s1, c1 in d1.items():
        target = -s1
        idx = bisect.bisect_left(keys2, target)
        
        # Check neighbors: idx-1, idx, idx+1
        # We need s1 + s2 != 0
        
        # Candidate 1: keys2[idx-1]
        if idx > 0:
            s2 = keys2[idx - 1]
            total = s1 + s2
            if total != 0:
                abs_total = abs(total)
                if best_abs is None or abs_total < best_abs:
                    best_abs = abs_total
                    best_c1 = c1
                    best_c2 = d2[s2]
                    if best_abs == 1:
                        break
        
        # Candidate 2: keys2[idx]
        if idx < len(keys2):
            s2 = keys2[idx]
            total = s1 + s2
            if total != 0:
                abs_total = abs(total)
                if best_abs is None or abs_total < best_abs:
                    best_abs = abs_total
                    best_c1 = c1
                    best_c2 = d2[s2]
                    if best_abs == 1:
                        break
                    
        # Candidate 3: keys2[idx+1]
        if idx + 1 < len(keys2):
            s2 = keys2[idx + 1]
            total = s1 + s2
            if total != 0:
                abs_total = abs(total)
                if best_abs is None or abs_total < best_abs:
                    best_abs = abs_total
                    best_c1 = c1
                    best_c2 = d2[s2]
                    if best_abs == 1:
                        break

    if best_c1 is None:
        # Fallback: This should not happen
        return [0] * 28 + [1, -1]

    coeffs = list(best_c1) + list(best_c2)
    
    # Final verification
    total_frac = sum(Fraction(c, i + 1) for i, c in enumerate(coeffs))
    if total_frac == 0:
        # This is a bug in logic if it happens, but fallback just in case
        return [0] * 28 + [1, -1]
        
    return coeffs

# EVOLVE-BLOCK-END
