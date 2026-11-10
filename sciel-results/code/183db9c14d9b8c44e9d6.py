"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    best_window = float('inf')
    best_result = None
    
    # We'll try candidate lower bounds L.
    # The optimal L is likely a multiple of one of the divisors, or near one.
    # Let's try L from 1 to a reasonable bound, and also try L values that are
    # multiples of each divisor.
    
    # Strategy: For each L in a range, greedily assign multiples.
    # To handle distinctness, if a multiple is already taken, use the next one.
    
    # Let's try L from 1 to 50000 (should be sufficient for small windows)
    # Also try some larger L values that are multiples of divisors.
    
    def try_window(L):
        """Try to find distinct multiples >= L for each divisor, minimize max."""
        used = set()
        values = []
        for d in divisors:
            # Smallest multiple of d >= L
            if L % d == 0:
                m = L
            else:
                m = L + (d - L % d)
            # Ensure distinctness
            while m in used:
                m += d
            if m > max_val:
                return None, float('inf')
            used.add(m)
            values.append(m)
        window = max(values) - L
        return values, window
    
    # Try L from 1 to 20000
    for L in range(1, 20001):
        values, window = try_window(L)
        if values is not None and window < best_window:
            best_window = window
            best_result = values
    
    # Also try L values that are multiples of each divisor, up to 100000
    for d in divisors:
        for L in range(d, 100001, d):
            if L > 20000:  # Only try ones not already covered
                values, window = try_window(L)
                if values is not None and window < best_window:
                    best_window = window
                    best_result = values
    
    # Convert to Python ints
    result = [int(x) for x in best_result]
    return result
# EVOLVE-BLOCK-END
