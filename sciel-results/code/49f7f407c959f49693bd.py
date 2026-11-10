"""Improved program for distinct three-term unit-fraction representations of 4/n."""
import time

# EVOLVE-BLOCK-START
def construct():
    MAX_DENOM = 10**12
    deadline = time.time() + 85  # safety margin before 90s kill
    answers = [None] * 254  # indices 0..253 correspond to n=3..256
    
    # Precompute: for each n, we need x, y, z distinct positive integers <= MAX_DENOM
    # such that 1/x + 1/y + 1/z = 4/n
    # Assume x < y < z (we'll search in this order and ensure distinctness)
    # From 1/x + 1/y + 1/z = 4/n:
    #   1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
    #   Let a = 4x - n, b = nx
    #   Then 1/y + 1/z = a/b
    #   (y + z) / (yz) = a/b
    #   b(y + z) = a*y*z
    #   a*y*z - b*y - b*z = 0
    #   Multiply by a: a^2*y*z - a*b*y - a*b*z = 0
    #   (a*y - b)(a*z - b) = b^2
    # So if we let u = a*y - b and v = a*z - b, then u*v = b^2
    # and y = (u + b) / a, z = (v + b) / a
    # We need u and v to be positive divisors of b^2, with u < v (since y < z)
    # and (u + b) % a == 0 and (v + b) % a == 0
    # Also y > x and z > y, and all distinct.
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        # x must satisfy: 4x > n (so a > 0), and x < n/4 * something...
        # Since x < y < z, we have 3/x > 4/n => x < 3n/4
        # Also 1/x < 4/n => x > n/4
        # So x is in (n/4, 3n/4)
        x_min = n // 4 + 1
        x_max = (3 * n) // 4  # exclusive upper bound for x < 3n/4
        
        # Try small x values first (likely to yield smaller y, z)
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n
            b = n * x
            if a <= 0:
                continue
            # Find divisors u of b^2 such that:
            #   u < b (so that y = (u+b)/a < (b+b)/a = 2b/a, and we need y > x)
            #   (u + b) % a == 0
            #   v = b^2 / u > u (so z > y)
            #   (v + b) % a == 0
            #   y = (u + b) // a > x
            #   z = (v + b) // a > y
            #   y, z <= MAX_DENOM
            #   x, y, z all distinct (ensured by x < y < z)
            
            # b^2 can be huge. Instead, iterate over divisors of b^2.
            # But b can be up to ~256 * 192 = ~49152, so b^2 ~ 2.4e9. 
            # Factoring b^2 and generating divisors is feasible.
            
            # Actually, let's be smarter. We can iterate u over divisors of b^2 
            # that are less than b. But b^2 has many divisors potentially.
            # Alternative: iterate y directly.
            # From 1/y + 1/z = a/b, with y > x:
            #   1/z = a/b - 1/y = (a*y - b) / (b*y)
            #   z = b*y / (a*y - b)
            # We need a*y - b > 0 => y > b/a
            # And z > y => b*y / (a*y - b) > y => b > a*y - b => 2b > a*y => y < 2b/a
            # So y is in (b/a, 2b/a)
            # And (a*y - b) must divide b*y.
            
            y_min = b // a + 1
            if y_min <= x:
                y_min = x + 1
            y_max = (2 * b) // a  # y < 2b/a, so y <= 2b/a - 1 effectively, but use floor
            
            # If y_max < y_min, no solution for this x
            if y_max < y_min:
                continue
            
            # We can't iterate all y if the range is large. 
            # For small n, the range is manageable. For larger n, it might be large.
            # Let's check the range size.
            range_size = y_max - y_min + 1
            if range_size > 100000:
                # Skip this x if the range is too large; try other x values
                continue
            
            for y in range(y_min, y_max + 1):
                denom = a * y - b
                if denom <= 0:
                    continue
                num = b * y
                if num % denom != 0:
                    continue
                z = num // denom
                if z <= y:
                    continue
                if z > MAX_DENOM:
                    continue
                if y > MAX_DENOM:
                    continue
                # Check distinctness: x < y < z ensures distinct
                found = [x, y, z]
                break
            if found:
                break
        
        answers[idx] = found
    
    return answers
# EVOLVE-BLOCK-END
