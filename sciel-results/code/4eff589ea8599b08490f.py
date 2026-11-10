import time
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Precompute some useful constants
    # LCM of all divisors
    def lcm(a, b):
        return a * b // gcd(a, b)
    LCM = reduce(lcm, divisors)  # 30030
    
    # We need to find 10 distinct multiples, one for each divisor,
    # minimizing max(reps) - min(reps).
    #
    # Strategy: Enumerate window sizes W starting from 9 (minimum for 10 distinct ints).
    # For each W, enumerate all possible window start positions modulo LCM.
    # For a window [start, start+W], check if we can assign distinct multiples.
    #
    # Optimization: Use backtracking with most-constrained-first ordering.
    # The number of multiples of d in a window of size W is at most W//d + 1,
    # so larger divisors have fewer choices.
    
    start_time = time.time()
    timeout = 80  # Safety margin for 90s limit
    
    # Pre-sort indices by divisor size (descending) for backtracking
    # Larger divisors have fewer multiples in any window, so they're more constrained
    indices_by_constraint = sorted(range(n), key=lambda i: -divisors[i])
    
    for W in range(9, 300):
        if time.time() - start_time > timeout:
            break
            
        # For window size W, the start position only matters modulo LCM
        # because the pattern of multiples repeats every LCM
        for L in range(1, LCM + 1):
            if time.time() - start_time > timeout:
                break
                
            lo = L
            hi = L + W
            
            # Collect available multiples for each divisor in [lo, hi]
            available = [None] * n
            valid = True
            
            for i in range(n):
                d = divisors[i]
                k = (lo + d - 1) // d  # smallest multiple >= lo
                first = k * d
                if first > hi:
                    valid = False
                    break
                mults = []
                x = first
                while x <= hi:
                    mults.append(x)
                    x += d
                available[i] = mults
            
            if not valid:
                continue
            
            # Check if we can assign distinct values using backtracking
            # Use bitmask for efficiency since values are in a small range
            # Map values to bit indices: val -> val - lo
            used_mask = 0
            choices = [0] * n
            found = False
            
            # Backtrack in order of most constrained first
            def backtrack(pos):
                nonlocal found
                if found:
                    return
                if pos == n:
                    found = True
                    return
                i = indices_by_constraint[pos]
                for val in available[i]:
                    bit = 1 << (val - lo)
                    if used_mask & bit:
                        continue
                    used_mask |= bit
                    choices[i] = val
                    backtrack(pos + 1)
                    if found:
                        return
                    used_mask &= ~bit
            
            backtrack(0)
            
            if found:
                result = [choices[i] for i in range(n)]
                # Verify
                if len(set(result)) == n:
                    for i in range(n):
                        if result[i] % divisors[i] != 0 or not (0 < result[i] <= 1000000):
                            break
                    else:
                        return result
    
    # Fallback: return the divisors themselves (valid but not optimal)
    return divisors[:]
