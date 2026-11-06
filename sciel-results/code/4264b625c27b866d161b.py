"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    results = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        
        # We need 1/x + 1/y + 1/z = 4/n with x < y < z (distinct positive integers)
        # Rearranging: 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
        # Let a = 4x - n, b = n*x
        # Then 1/y + 1/z = a/b
        # => (y+z)/(yz) = a/b
        # => b(y+z) = a*y*z
        # => a*y*z - b*y - b*z = 0
        # Multiply by a: a^2*y*z - a*b*y - a*b*z = 0
        # Add b^2: (a*y - b)(a*z - b) = b^2
        # So for each divisor d of b^2, we can set:
        #   a*y - b = d  =>  y = (d + b) / a
        #   a*z - b = b^2/d  =>  z = (b^2/d + b) / a
        # We need d + b ≡ 0 (mod a) and b^2/d + b ≡ 0 (mod a)
        # Also need y, z to be positive integers, distinct from each other and from x, with y < z.
        
        # x must satisfy 4x > n (so a > 0), and x > n/4.
        # Also, since 1/x < 4/n (because 1/y + 1/z > 0), we need x > n/4.
        # And 1/x > 4/n - 2/y_min... but let's just iterate x from floor(n/4)+1 upward.
        # Upper bound on x: since 1/y + 1/z >= 2/(y+z) and y > x, z > y, 
        # actually the maximum 1/x can be is when y and z are as large as possible,
        # but we need 1/x < 4/n. Also 1/x > 4/n - 2/(x+1) approximately.
        # A simple upper bound: x < n (since 1/x + 1/y + 1/z < 3/x <= 4/n => x > 3n/4... wait)
        # Actually 1/x < 4/n => x > n/4. And 1/x > 4/n - 2/(x+1) is not a clean bound.
        # Let's use: since y > x and z > y > x, we have 1/y < 1/x and 1/z < 1/x,
        # so 4/n = 1/x + 1/y + 1/z < 3/x, hence x < 3n/4.
        # Wait, that's not right either. 1/y < 1/x means y > x. So 1/x + 1/y + 1/z < 1/x + 1/x + 1/x = 3/x.
        # So 4/n < 3/x => x < 3n/4.
        # Also 1/x < 4/n => x > n/4.
        # So x ranges from floor(n/4) + 1 to floor(3n/4) - 1 (or 3n/4 if not integer).
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4  # inclusive upper bound
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n
            b = n * x
            if a <= 0:
                continue
            
            # We need to find divisors d of b^2 such that:
            # (d + b) % a == 0 and (b*b//d + b) % a == 0
            # and y = (d+b)//a > x, z = (b*b//d + b)//a > y, all distinct.
            
            # To avoid iterating over all divisors of b^2 (which can be huge),
            # let's use a different approach: iterate over possible y values.
            # From 1/y + 1/z = a/b, we have z = b*y / (a*y - b).
            # We need a*y - b > 0, so y > b/a.
            # Also z > y means b*y / (a*y - b) > y => b > a*y - b => 2b > a*y => y < 2b/a.
            # So y ranges from floor(b/a) + 1 to floor(2b/a) - 1.
            # And we need (a*y - b) divides b*y.
            # Note gcd(a*y - b, a*y) = gcd(a*y - b, b) since gcd(a*y - b, a*y) = gcd(a*y - b, a*y - (a*y - b)) = gcd(a*y - b, b).
            # So let g = gcd(a*y - b, b). Then (a*y - b)/g must divide y.
            
            y_min = b // a + 1
            y_max = (2 * b) // a - 1
            
            # Ensure y_min > x (since denominators must be distinct and we want x < y < z)
            if y_min <= x:
                y_min = x + 1
            
            if y_min > y_max:
                continue
            
            # Iterate y in range
            for y in range(y_min, y_max + 1):
                if time.time() > deadline:
                    break
                denom = a * y - b
                if denom <= 0:
                    continue
                # Check if denom divides b*y
                if (b * y) % denom == 0:
                    z = (b * y) // denom
                    if z > y and z != x and y != x:
                        # Verify all distinct and positive
                        if x != y and y != z and x != z:
                            # Verify the equation
                            # 1/x + 1/y + 1/z == 4/n
                            # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
                            if n * (y * z + x * z + x * y) == 4 * x * y * z:
                                if x <= 1000000000000 and y <= 1000000000000 and z <= 1000000000000:
                                    found = [x, y, z]
                                    break
            if found:
                break
        
        results[idx] = found
    
    return results
# EVOLVE-BLOCK-END
