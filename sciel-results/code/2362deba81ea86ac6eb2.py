import numpy as np
from math import gcd
from functools import reduce
import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 90s kill
    
    n = 30
    def lcm(a, b):
        return a // gcd(a, b) * b
    
    L = reduce(lcm, range(1, n + 1))
    weights = [L // (i + 1) for i in range(n)]
    
    half = n // 2
    w1 = weights[:half]
    w2 = weights[half:]
    
    # Enumerate sums and track one coefficient assignment per sum
    # Using a dict: sum -> coefficients tuple
    def enumerate_half(w):
        """Generate dict mapping sum -> coeffs tuple for given weights."""
        d = {0: tuple([0] * len(w))}
        for pos, wi in enumerate(w):
            new_d = {}
            for s, c in d.items():
                # coeff = 0
                key = s
                if key not in new_d:
                    new_d[key] = c[:pos] + (0,) + c[pos+1:]
                # coeff = 1
                key = s + wi
                if key not in new_d:
                    new_d[key] = c[:pos] + (1,) + c[pos+1:]
                # coeff = -1
                key = s - wi
                if key not in new_d:
                    new_d[key] = c[:pos] + (-1,) + c[pos+1:]
            d = new_d
        return d
    
    # Check time
    if time.time() > deadline:
        return [0]*28 + [1, -1]
    
    # Enumerate first half
    d1 = enumerate_half(w1)
    
    if time.time() > deadline:
        return [0]*28 + [1, -1]
    
    # Enumerate second half
    d2 = enumerate_half(w2)
    
    if time.time() > deadline:
        return [0]*28 + [1, -1]
    
    # Sort keys of d2 for binary search
    keys2 = sorted(d2.keys())
    
    best_abs = None
    best_s1 = 0
    best_s2 = 0
    best_c1 = None
    best_c2 = None
    
    # For each s1 in d1, find s2 in d2 such that s1+s2 is closest to 0 but not 0
    for s1, c1 in d1.items():
        target = -s1
        idx = bisect.bisect_left(keys2, target)
        
        # Check candidates around idx: idx-1, idx, idx+1
        for i in [idx - 1, idx, idx + 1]:
            if 0 <= i < len(keys2):
                s2 = keys2[i]
                total = s1 + s2
                if total != 0:
                    abs_total = abs(total)
                    if best_abs is None or abs_total < best_abs:
                        best_abs = abs_total
                        best_s1 = s1
                        best_s2 = s2
                        best_c1 = c1
                        best_c2 = d2[s2]
    
    if best_c1 is None or best_c2 is None:
        # Fallback
        return [0]*28 + [1, -1]
    
    # Combine coefficients
    coeffs = list(best_c1) + list(best_c2)
    
    # Verify
    assert len(coeffs) == 30
    assert all(c in (-1, 0, 1) for c in coeffs)
    
    # Verify the sum is nonzero
    from fractions import Fraction
    total_frac = sum(Fraction(c, i + 1) for i, c in enumerate(coeffs))
    assert total_frac != 0, "Sum must be nonzero"
    
    return coeffs
