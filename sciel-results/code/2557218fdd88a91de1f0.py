"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    n = 30
    # LCM(1..30) for exact integer arithmetic
    # LCM(1..30) = 2329089562800
    # But we can work with Fractions or use a common denominator
    
    # Meet-in-the-middle: split into first 15 and last 15
    half = 15
    
    # Generate all 3^15 assignments for first half (indices 0..14, terms 1/1 .. 1/15)
    # and second half (indices 15..29, terms 1/16 .. 1/30)
    
    def generate_half(start_idx, end_idx):
        """Generate all (sum_as_fraction, coeffs) for terms start_idx to end_idx (inclusive)"""
        results = []
        num_terms = end_idx - start_idx + 1
        # Use iterative generation
        coeffs = [0] * num_terms
        for mask in range(3 ** num_terms):
            m = mask
            c = [0] * num_terms
            for i in range(num_terms):
                c[i] = m % 3 - 1  # maps 0,1,2 to -1,0,1
                m //= 3
            # Compute sum
            s = Fraction(0, 1)
            for i in range(num_terms):
                if c[i] != 0:
                    s += Fraction(c[i], start_idx + i + 1)
            results.append((s, c))
        return results
    
    # This is too slow for 3^15 = 14348907 entries with Fraction
    # Let's use integer arithmetic with a common denominator for each half
    
    # For first half: terms 1/1 to 1/15, LCM(1..15) = 360360
    # For second half: terms 1/16 to 1/30, LCM(16..30)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # LCM(1..15)
    lcm1 = 1
    for i in range(1, 16):
        lcm1 = lcm(lcm1, i)
    # LCM(16..30)
    lcm2 = 1
    for i in range(16, 31):
        lcm2 = lcm(lcm2, i)
    
    # Common denominator for all: LCM(1..30)
    lcm_all = lcm(lcm1, lcm2)
    
    # Precompute weights for each term as integer: weight[i] = lcm_all // (i+1)
    weights = [lcm_all // (i + 1) for i in range(n)]
    
    # First half: indices 0..14, weights[0..14]
    # Second half: indices 15..29, weights[15..29]
    
    def gen_half_ints(w_start, w_end):
        """Generate all (integer_sum, coeffs_list) for weights w_start to w_end"""
        num_terms = w_end - w_start + 1
        results = []
        # Iterative generation
        coeffs = [0] * num_terms
        for mask in range(3 ** num_terms):
            m = mask
            c = [0] * num_terms
            for i in range(num_terms):
                c[i] = m % 3 - 1
                m //= 3
            s = 0
            for i in range(num_terms):
                if c[i] != 0:
                    s += c[i] * weights[w_start + i]
            results.append((s, c))
        return results
    
    # Check time
    if time.time() > deadline - 40:
        # Fallback to simple heuristic
        return [0] * 28 + [1, -1]
    
    # Generate first half
    half1 = gen_half_ints(0, 14)
    
    if time.time() > deadline - 30:
        return [0] * 28 + [1, -1]
    
    # Generate second half
    half2 = gen_half_ints(15, 29)
    
    if time.time() > deadline - 20:
        return [0] * 28 + [1, -1]
    
    # Sort half2 by sum value
    half2.sort(key=lambda x: x[0])
    
    # For each entry in half1, find the best match in half2
    # We want s1 + s2 != 0, minimize |s1 + s2|
    # Binary search for -s1 in half2's sorted sums
    
    import bisect
    
    half2_sums = [x[0] for x in half2]
    
    best_abs = None
    best_coeffs = None
    
    for s1, c1 in half1:
        # We want s2 such that s1 + s2 is close to 0 but not 0
        # Search around -s1
        target = -s1
        pos = bisect.bisect_left(half2_sums, target)
        
        # Check positions around pos
        for offset in range(-5, 6):
            idx = pos + offset
            if idx < 0 or idx >= len(half2_sums):
                continue
            s2 = half2_sums[idx]
            total = s1 + s2
            if total == 0:
                continue
            abs_total = abs(total)
            if best_abs is None or abs_total < best_abs:
                best_abs = abs_total
                c2 = half2[idx][1]
                best_coeffs = c1 + c2
                if best_abs == 1:
                    # Can't do better than 1 (in units of 1/lcm_all)
                    break
        if best_abs == 1:
            break
    
    if best_coeffs is None:
        # Fallback
        return [0] * 28 + [1, -1]
    
    # Verify the result
    total_frac = Fraction(0, 1)
    for i in range(n):
        if best_coeffs[i] != 0:
            total_frac += Fraction(best_coeffs[i], i + 1)
    
    if total_frac == 0:
        # Shouldn't happen, but fallback
        return [0] * 28 + [1, -1]
    
    return best_coeffs
# EVOLVE-BLOCK-END

