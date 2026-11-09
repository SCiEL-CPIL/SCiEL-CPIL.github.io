"""Rewritten program with a stronger divisor-enumeration strategy."""
# EVOLVE-BLOCK-START
import math

def _divisors(num):
    """Return list of all positive divisors of num."""
    if num == 0:
        return [0]
    num = abs(num)
    divs = []
    i = 1
    while i * i <= num:
        if num % i == 0:
            divs.append(i)
            if i != num // i:
                divs.append(num // i)
        i += 1
    return divs

def construct():
    answers = []
    BOUND = 1000000000000  # 10^12
    
    for n in range(3, 257):
        found = None
        # We need 1/x + 1/y + 1/z = 4/n with x < y < z, all distinct positive integers <= BOUND
        # Strategy: iterate over possible x values. For each x, we have:
        #   1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
        # Let a = 4x - n, b = nx. We need a > 0, so x > n/4.
        #   1/y + 1/z = a/b  =>  b(y+z) = az yz  =>  az yz - by - bz = 0
        #   (az y - b)(az z - b) = b^2
        # So we need to factor b^2 and find pairs.
        # Alternatively, iterate over divisors of b^2... but b^2 can be huge.
        # Better: iterate over divisors d of b such that... 
        # Actually, from 1/y + 1/z = a/b:
        #   y+z = (a/b)yz
        #   b(y+z) = a y z
        #   a y z - b y - b z = 0
        #   (a y - b)(a z - b) = b^2
        # Let u = a y - b, v = a z - b, then u*v = b^2, and y = (u+b)/a, z = (v+b)/a
        # We need y, z to be positive integers, so u+b ≡ 0 (mod a) and v+b ≡ 0 (mod a)
        # Also y > x, z > y, and all <= BOUND.
        
        # For efficiency, we try small x first (starting from floor(n/4)+1)
        # and for each x, we enumerate divisors of b^2... but that's expensive.
        # Instead, we can iterate over divisors u of b^2 such that u <= b^2 (and v = b^2/u)
        # But b^2 can be up to (n*10^6)^2 which is too big.
        
        # Alternative: iterate over possible y values? No, that's what the baseline does.
        
        # Better approach: For a given x, a = 4x - n, b = n*x.
        # We need u * v = b^2 where u ≡ -b (mod a) and v ≡ -b (mod a).
        # Since u*v = b^2 and u ≡ v ≡ -b (mod a), we have b^2 ≡ b^2 (mod a), which is consistent.
        # We can iterate over divisors u of b^2. But b^2 is too large.
        
        # Let's use the fact that we only need to find ONE solution per n.
        # Try small x values first. For each x, check if a solution exists by
        # iterating over a limited range of divisors or using a smarter search.
        
        # Actually, let's try a different parameterization.
        # Known identity: if n is odd, 4/n = 1/((n+1)/2) + 1/((n+1)/2) + 1/(2n(n+1)) -- but denominators not distinct.
        # For distinct: we can use the method of trying small x.
        
        # Let's try x from floor(n/4)+1 upward, but limit the search.
        # For each x, compute a = 4x - n, b = n*x.
        # We need to find y > x such that (a*y - b) divides b^2 and z = b^2/(a*y-b) + b/a... 
        # Wait: (a*y - b)(a*z - b) = b^2
        # So a*y - b must be a divisor of b^2.
        # Let d = a*y - b, so d | b^2, and y = (d + b) / a must be integer > x.
        # Then z = (b^2/d + b) / a must be integer > y.
        
        # To avoid enumerating all divisors of b^2 (which is too many), we can:
        # - Try a few small x values
        # - For each x, try a limited number of divisors d of b^2
        
        # But even finding divisors of b^2 is expensive for large b.
        
        # Alternative: use the fact that we just need one solution.
        # Try x = ceil(n/4) + k for small k.
        
        best_found = None
        # Try x in a reasonable range
        x_start = n // 4 + 1
        # Limit x search to keep within time
        for x in range(x_start, min(x_start + 200, 1000000)):
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            # We need d | b^2, d ≡ -b (mod a), y = (d+b)/a > x, z = (b^2/d + b)/a > y
            # Instead of enumerating all divisors of b^2, let's try to find d by iterating
            # over a small set of candidate divisors.
            
            # Actually, let's use a different tactic: iterate over possible y in a limited range
            # for this x. This is similar to the baseline but with a wider x range.
            # For given x, a, b:
            #   1/y + 1/z = a/b
            #   z = b*y / (a*y - b)
            # We need a*y - b > 0 => y > b/a
            # And z > y => b*y/(a*y-b) > y => b > a*y - b => 2b > a*y => y < 2b/a
            # So y in (b/a, 2b/a)
            y_min = b // a + 1
            y_max = min(2 * b // a, BOUND)
            # Limit the search range to avoid timeout
            if y_max - y_min > 100000:
                # Sample or limit
                y_max = y_min + 100000
            
            for y in range(y_min, y_max + 1):
                denom = a * y - b
                if denom <= 0:
                    continue
                if (b * y) % denom != 0:
                    continue
                z = (b * y) // denom
                if z <= y:
                    continue
                if z > BOUND:
                    continue
                # Check distinctness
                if x != y and y != z and x != z:
                    # Verify the equation
                    # 1/x + 1/y + 1/z = 4/n
                    # Cross multiply: n*(y*z + x*z + x*y) = 4*x*y*z
                    lhs = n * (y * z + x * z + x * y)
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        found = [x, y, z]
                        break
            if found:
                break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
