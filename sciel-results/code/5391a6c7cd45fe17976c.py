"""Rewritten program for distinct three-term unit-fraction representations of 4/n."""
import time
import math
from math import gcd

def construct():
    deadline = time.time() + 85
    results = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        
        # 1/x + 1/y + 1/z = 4/n with x < y < z distinct positive integers
        # For each x in (n/4, 3n/4):
        #   A = 4x - n, B = n*x
        #   Need 1/y + 1/z = A/B
        #   => (A*y - B)(A*z - B) = B^2
        #   => z = B*y / (A*y - B), need integer z > y
        #
        # Alternative: iterate over divisors of B^2.
        # Let d = A*y - B, then A*z - B = B^2/d
        # y = (d + B)/A, z = (B^2/d + B)/A
        # Need d | B^2, d < B (since y < z), and A | (d + B), A | (B^2/d + B)
        #
        # For efficiency, iterate y directly but use the divisor structure.
        # Actually, let's use the direct y-iteration approach but with a tighter bound.
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Try x values; start from smaller x (closer to n/4) which tends to give
        # smaller A and thus potentially easier to find y, z
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # y must satisfy: y > B/A and y > x, and y < 2B/A
            y_min = max(x + 1, B // A + 1)
            y_max = 2 * B // A
            
            if y_max < y_min:
                continue
            
            # Iterate y, checking if B*y is divisible by (A*y - B)
            for y in range(y_min, y_max + 1):
                if time.time() > deadline:
                    break
                denom = A * y - B
                if denom <= 0:
                    continue
                if (B * y) % denom != 0:
                    continue
                z = (B * y) // denom
                if z > y and z <= 10**12:
                    found = [x, y, z]
                    break
            if found:
                break
        
        results[idx] = found
    
    return results
