import time
import math

def construct():
    deadline = time.time() + 78.0
    answers = [None] * 254
    BOUND = 10**12
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        
        found = None
        
        # Try k=1,2,3 where x = n/k
        for k in [1, 2, 3]:
            if time.time() > deadline:
                break
            if n % k != 0:
                continue
            x = n // k
            if x <= 0 or x > BOUND:
                continue
            
            # Need 1/y + 1/z = (4-k)/n
            m = 4 - k
            if m <= 0:
                continue
            
            # (my - n)(mz - n) = n^2
            # Let u = my - n, v = mz - n, then uv = n^2
            # y = (u + n)/m, z = (v + n)/m
            # Need u, v positive divisors of n^2 with u*v = n^2
            # And (u+n) % m == 0, (v+n) % m == 0
            # And y, z distinct from x and each other, all positive
            
            n2 = n * n
            
            # Enumerate divisors u of n^2 where u <= n^2/u (to avoid duplicates)
            # u must satisfy: u > 0, (u + n) % m == 0
            # v = n^2 / u, (v + n) % m == 0
            
            # Find all divisors of n^2
            # n is at most 256, so n^2 is at most 65536, manageable
            divisors = []
            for d in range(1, int(math.isqrt(n2)) + 1):
                if n2 % d == 0:
                    divisors.append(d)
                    if d != n2 // d:
                        divisors.append(n2 // d)
            
            for u in divisors:
                if time.time() > deadline:
                    break
                v = n2 // u
                # Check divisibility conditions
                if (u + n) % m != 0:
                    continue
                if (v + n) % m != 0:
                    continue
                
                y = (u + n) // m
                z = (v + n) // m
                
                # Check positivity and bounds
                if y <= 0 or z <= 0:
                    continue
                if y > BOUND or z > BOUND:
                    continue
                
                # Check distinctness: x, y, z all distinct
                if y == x or z == x or y == z:
                    continue
                
                # Ensure y, z are positive integers (already checked)
                # Verify the equation
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    # Sort for consistency, but we just need them distinct
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
