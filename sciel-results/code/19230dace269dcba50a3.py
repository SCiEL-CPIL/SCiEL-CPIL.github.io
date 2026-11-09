import time
import math

def construct():
    answers = []
    deadline = time.time() + 55  # Safety margin before 60s limit
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must satisfy n/4 < x < 3n/4 for distinct positive solutions with x < y < z
        x_start = n // 4 + 1
        # x < 3n/4, so x <= floor(3n/4 - epsilon), use 3n//4 as upper bound
        # But we need 3/x > 4/n strictly for y,z > x to be possible
        # Actually 1/y + 1/z < 2/x (since y,z > x), so 4/n - 1/x < 2/x => 4/n < 3/x => x < 3n/4
        x_end = (3 * n) // 4  # inclusive upper bound for x
        
        for x in range(x_start, x_end + 1):
            if time.time() > deadline:
                break
            
            a = 4 * x - n  # numerator of (4/n - 1/x)
            b = n * x       # denominator of (4/n - 1/x)
            
            if a <= 0:
                continue
            
            # We need 1/y + 1/z = a/b with y > x, z > y
            # (a*y - b)(a*z - b) = b^2
            # Let d = a*y - b, then d divides b^2, d < b (since y > b/a means d > 0)
            # Actually: a*y - b > 0 requires y > b/a
            # And a*z - b = b^2/d, so z = (b^2/d + b)/a
            # We need y > x and z > y and all distinct
            
            # y = (d + b) / a must be integer, so (d + b) % a == 0
            # z = (b^2/d + b) / a must be integer, so (b^2//d + b) % a == 0
            # Also need d < b^2/d (i.e., d < b) for y < z, but let's just check y < z
            
            # Iterate over divisors d of b^2 where d < b
            # b can be large (up to 256 * ~192 = ~49152), b^2 up to ~2.4 * 10^9
            # Finding all divisors of b^2 might be expensive for large b
            
            # Alternative: iterate y from floor(b/a) + 1 upward
            # 1/y + 1/z = a/b => z = b*y / (a*y - b)
            # Need a*y - b > 0, so y > b/a
            # Need z > y, so b*y / (a*y - b) > y => b > a*y - b => 2*b > a*y => y < 2*b/a
            
            y_min = b // a + 1
            y_max = (2 * b) // a  # y < 2*b/a
            
            # Also need y > x for distinctness (x < y < z)
            if y_min <= x:
                y_min = x + 1
            
            if y_min > y_max:
                continue
            
            # Additionally, z = b*y / (a*y - b) must be integer and > y
            # And z <= 10^12
            # Let's iterate y in this range
            for y in range(y_min, y_max + 1):
                denom = a * y - b
                if denom <= 0:
                    continue
                if (b * y) % denom != 0:
                    continue
                z = (b * y) // denom
                if z <= y:
                    continue
                if z > 10**12:
                    continue
                # Check distinctness: x, y, z all distinct
                if x != y and y != z and x != z:
                    # Verify: 1/x + 1/y + 1/z == 4/n
                    # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
                    lhs = n * (y * z + x * z + x * y)
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
