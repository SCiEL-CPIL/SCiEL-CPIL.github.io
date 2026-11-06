"""Optimized constructor for distinct three-term unit-fraction reps of 4/n."""
import time
import math
from math import gcd

DENOM_BOUND = 10**12

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254  # index 0 -> n=3, ..., index 253 -> n=256
    
    for i, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        
        found = None
        x_min = n // 4 + 1
        
        # Strategy: prioritize x values that give small y-search ranges.
        # The y-range length is ~ n*x/(4x-n). For large x, this approaches n/4.
        # For x close to n/4, it's huge. So try larger x first.
        
        # Build candidate list: mix of strategic values and a sweep
        # For each x, the y-range is [floor(nx/(4x-n))+1, floor(2nx/(4x-n))]
        # Range length ≈ nx/(4x-n)
        
        x_candidates = set()
        
        # Add strategic x values: multiples and near-multiples of n
        for mult in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
            x_val = mult * n
            if x_val > x_min:
                x_candidates.add(x_val)
        
        # Add x values near n/4 (small A, but might work)
        for offset in range(1, 50):
            x_val = x_min + offset
            x_candidates.add(x_val)
        
        # Add some mid-range values
        for frac_num in range(1, 20):
            x_val = (frac_num * n) // 4 + 1
            if x_val > x_min and x_val < 10 * n:
                x_candidates.add(x_val)
        
        # Sort by expected y-range size (smaller range = faster to search)
        # y_range ≈ n*x/(4x-n)
        def y_range_estimate(x):
            A = 4 * x - n
            if A <= 0:
                return float('inf')
            return n * x / A
        
        x_candidates = sorted(x_candidates, key=y_range_estimate)
        
        # Cap the number of x candidates to avoid spending too much time
        max_x_candidates = 200
        if len(x_candidates) > max_x_candidates:
            x_candidates = x_candidates[:max_x_candidates]
        
        for x in x_candidates:
            if time.time() > deadline:
                break
            if x <= x_min - 1:
                continue
            if x > DENOM_BOUND:
                continue
                
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # Need 1/y + 1/z = A/B
            # y must satisfy: B/A < y <= 2B/A (assuming y <= z)
            y_lo = B // A + 1
            y_hi = (2 * B) // A
            
            if y_hi < y_lo:
                continue
            
            # Ensure y > x for ordering (we want distinct, so assume x < y < z)
            if y_lo <= x:
                y_lo = x + 1
            if y_lo > y_hi:
                continue
            
            # Limit the search range for y
            max_y_iter = 50000
            y_max_check = min(y_hi, y_lo + max_y_iter - 1)
            
            for y in range(y_lo, y_max_check + 1):
                if time.time() > deadline:
                    break
                    
                denom = A * y - B
                if denom <= 0:
                    continue
                
                # Check if B*y is divisible by denom
                if (B * y) % denom == 0:
                    z = (B * y) // denom
                    
                    # Check z > y (for ordering) and all distinct and within bounds
                    if z > y and z != x and z <= DENOM_BOUND:
                        # Verify distinctness
                        if x != y and y != z and x != z:
                            # Verify the equation: 1/x + 1/y + 1/z = 4/n
                            # Cross multiply: n*(y*z + x*z + x*y) = 4*x*y*z
                            lhs = n * (y * z + x * z + x * y)
                            rhs = 4 * x * y * z
                            if lhs == rhs:
                                found = [x, y, z]
                                break
            
            if found:
                break
        
        answers[i] = found
    
    return answers
