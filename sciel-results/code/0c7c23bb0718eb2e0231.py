"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    deadline = time.time() + 85
    results = [None] * 254
    BOUND = 10**12
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        
        x_lo = n // 4 + 1
        x_hi = (3 * n + 3) // 4
        
        for x in range(x_lo, x_hi + 1):
            a = 4 * x - n
            b = n * x
            if a <= 0:
                continue
            
            # We need: 1/y + 1/z = (4x - n)/(nx) = a/b
            # So: (y+z)/(yz) = a/b => b(y+z) = a*yz
            # Rearranging: a*yz - b*y - b*z = 0
            # Multiply by a: a²*yz - ab*y - ab*z = 0
            # Add b²: (ay-b)(az-b) = b²
            # So we need d | b² where d = ay - b, and then z = (b²/d + b)/a
            
            # y range: b/a < y < 2*b/a
            y_lo = b // a + 1
            y_hi = min(2 * b // a, BOUND)
            
            # Also y > x for distinctness
            if y_lo <= x:
                y_lo = x + 1
            
            if y_lo > y_hi:
                continue
            
            # Find divisors d of b² such that y = (d + b)/a is in [y_lo, y_hi]
            # and z = (b²/d + b)/a > y and z <= BOUND
            # d must be a divisor of b²
            b_sq = b * b
            
            # Enumerate divisors of b_sq up to a certain limit
            # d = a*y - b, so d ranges from a*y_lo - b to a*y_hi - b
            d_lo = a * y_lo - b
            d_hi = a * y_hi - b
            
            found_for_x = None
            
            # Try small x first (likely to find solutions faster)
            # For efficiency, try a few y values near the boundaries
            # First try y near y_lo
            for y in range(y_lo, min(y_lo + 1000, y_hi + 1)):
                denom = a * y - b
                if denom <= 0:
                    continue
                num = b * y
                if num % denom == 0:
                    z = num // denom
                    if z > y and z <= BOUND:
                        if x != y and y != z and x != z:
                            found_for_x = [x, y, z]
                            break
            
            if not found_for_x:
                # Try y near y_hi
                for y in range(max(y_lo, y_hi - 1000), y_hi + 1):
                    denom = a * y - b
                    if denom <= 0:
                        continue
                    num = b * y
                    if num % denom == 0:
                        z = num // denom
                        if z > y and z <= BOUND:
                            if x != y and y != z and x != z:
                                found_for_x = [x, y, z]
                                break
            
            if not found_for_x and time.time() < deadline - 5:
                # Full search if needed
                for y in range(y_lo, y_hi + 1):
                    denom = a * y - b
                    if denom <= 0:
                        continue
                    num = b * y
                    if num % denom == 0:
                        z = num // denom
                        if z > y and z <= BOUND:
                            if x != y and y != z and x != z:
                                found_for_x = [x, y, z]
                                break
            
            if found_for_x:
                found = found_for_x
                break
        
        results[idx] = found
    
    return results
# EVOLVE-BLOCK-END
