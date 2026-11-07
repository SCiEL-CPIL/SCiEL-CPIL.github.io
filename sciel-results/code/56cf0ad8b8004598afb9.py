import math
import time

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = len(DIVISORS)
    
    # Calculate LCM of all divisors to determine the period
    lcm_val = 1
    for d in DIVISORS:
        lcm_val = _lcm(lcm_val, d)
    
    # The problem is periodic with period LCM.
    # We only need to search L in [1, LCM].
    # The "greedy" assignment (smallest multiple >= L) changes only when L crosses a multiple of some divisor.
    # Specifically, the set of values {ceil(L/d_i) * d_i} is constant on intervals ((m-1)*d_i, m*d_i].
    # The value changes at L = m*d_i + 1.
    # So we should check L = 1 and L = m*d_i + 1 for all multiples m*d_i < LCM.
    
    critical_Ls = set()
    critical_Ls.add(1)
    
    for d in DIVISORS:
        # Multiples are d, 2d, 3d, ...
        # The ceiling function ceil(L/d) changes value when L passes a multiple.
        # If L = k*d, ceil(L/d) = k.
        # If L = k*d + 1, ceil(L/d) = k+1.
        # So the "new" configuration starts at L = k*d + 1.
        # We must also consider the start of the period L=1.
        # Also, the configuration for L in [1, d] is determined by L=1.
        # The configuration for L in (d, 2d] is determined by L=d+1.
        # So we check L = 1 and L = k*d + 1 for k>=1 such that k*d + 1 <= LCM.
        
        for k in range(1, lcm_val // d + 1):
            L_candidate = k * d + 1
            if L_candidate <= lcm_val:
                critical_Ls.add(L_candidate)
            else:
                break
    
    # It is also possible that the optimal window starts exactly at a multiple?
    # If L = k*d, the smallest multiple >= L is k*d.
    # If L = k*d + 1, the smallest multiple >= L is (k+1)*d.
    # The set of candidates for L=k*d is {k*d for i where d_i|k*d, else next}.
    # The set of candidates for L=k*d-1 might be different?
    # Actually, checking L = k*d + 1 covers the "upper" side of the step.
    # What about L = k*d?
    # Let's include L = k*d as well to be safe, or just realize that the set of values
    # {ceil(L/d_i) * d_i} for L in [1, LCM] takes values only at these step boundaries.
    # The function f_i(L) = ceil(L/d_i)*d_i is a step function.
    # It is constant on intervals ( (m-1)d_i, m*d_i ].
    # The value on this interval is m*d_i.
    # The interval starts at (m-1)d_i + 1 and ends at m*d_i.
    # So we should pick a representative L from each interval.
    # L = (m-1)d_i + 1 is the start of the interval where the value is m*d_i.
    # So checking L = 1 and L = k*d_i + 1 is correct.
    
    critical_Ls = sorted(critical_Ls)
    
    best_span = MAX_VAL
    best_values = None
    
    # We iterate through all critical Ls
    for L in critical_Ls:
        if time.time() > deadline:
            break
            
        # Calculate the smallest multiple of each divisor >= L
        candidates = []
        for d in DIVISORS:
            # Smallest multiple of d >= L
            # q = ceil(L / d)
            q = (L + d - 1) // d
            val = q * d
            candidates.append((val, d))
        
        # To minimize the span, we want to assign the smallest possible distinct values.
        # Sort by the candidate value. If tie, sort by divisor (arbitrary but consistent).
        # Then greedily assign: if a value is already taken, increment by the divisor.
        candidates.sort(key=lambda x: (x[0], x[1]))
        
        used_values = set()
        assigned = []
        
        for val, d in candidates:
            v = val
            while v in used_values:
                v += d
                if v > MAX_VAL:
                    break
            if v > MAX_VAL:
                break
            used_values.add(v)
            assigned.append((v, d))
        
        if len(assigned) < n:
            continue
            
        # Calculate span
        min_val = min(v for v, _ in assigned)
        max_val = max(v for v, _ in assigned)
        span = max_val - min_val
        
        if span < best_span:
            best_span = span
            # We need to return the values in the order of the original DIVISORS list.
            # assigned contains (value, divisor). We need to map back.
            # Since divisors might have duplicates? No, DIVISORS are distinct.
            # But wait, DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39] - they are distinct.
            # So we can map divisor -> value.
            val_map = {d: v for v, d in assigned}
            best_values = [val_map[d] for d in DIVISORS]
            
            # Theoretical minimum span for 10 distinct integers is 9.
            # If we reach 9, we can stop early? 
            # It's possible to have span 9 if the numbers are consecutive.
            # Given the divisors, it's unlikely to be 9, but 10, 11 etc are possible.
            if best_span <= 10:
                # We can break if we find a very small span, but let's finish the loop 
                # unless it's extremely small to ensure we don't miss a smaller one? 
                # Actually, if we found span 9, it's the global minimum.
                if best_span == 9:
                    break

    if best_values is None:
        # Fallback: just use the divisors themselves (valid but not optimal)
        best_values = list(DIVISORS)
        
    return [int(v) for v in best_values]
