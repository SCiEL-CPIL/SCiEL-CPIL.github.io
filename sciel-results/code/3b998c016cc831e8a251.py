import time
import math
from math import gcd

def construct():
    """
    Construct 254 rows for n=3..256. Each row is None or three DISTINCT positive
    integer denominators x,y,z <= 10^12 with 1/x+1/y+1/z = 4/n.
    Maximize fraction solved.
    """
    MAX_DENOM = 10**12
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit
    
    answers = [None] * 254  # index 0 corresponds to n=3
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        
        # We need 1/x + 1/y + 1/z = 4/n with x < y < z, all distinct
        # x must be > n/4 (since 1/x < 4/n)
        # x must be <= 3n/4 (since 1/x >= 4/n - 2/n = 2/n... wait)
        # Actually: 1/x < 4/n => x > n/4
        # Also 1/x + 1/y + 1/z = 4/n, and y > x, z > y
        # So 1/x > 4/n - 2/x (since 1/y + 1/z < 2/x)
        # This gives 3/x > 4/n => x < 3n/4
        # So x is in range (n/4, 3n/4)
        
        # For each valid x, we need 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
        # Let A = 4x - n, B = nx
        # 1/y + 1/z = A/B
        # (y+z)/(yz) = A/B
        # B(y+z) = A*yz
        # A*yz - B*y - B*z = 0
        # (Ay - B)(Az - B) = B^2
        # So we need to factor B^2 = (nx)^2
        # Let d be a divisor of B^2, then Ay - B = d, Az - B = B^2/d
        # y = (d + B)/A, z = (B^2/d + B)/A
        # We need y, z to be positive integers, y > x, z > y, and all <= MAX_DENOM
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4  # x < 3n/4, so x <= floor(3n/4)
        
        # Try x values; for efficiency, try smaller x first (gives larger A, fewer divisors to check)
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
            
            # Need (Ay - B)(Az - B) = B^2
            # y = (d + B) / A must be integer, so (d + B) % A == 0
            # z = (B^2/d + B) / A must be integer, so (B^2/d + B) % A == 0
            # Also y > x, z > y, and all <= MAX_DENOM
            # d must be a positive divisor of B^2
            # Also Ay - B = d > 0 => y > B/A, so d > 0
            # And Az - B = B^2/d > 0 => always true for positive d
            
            # For y > x: (d + B)/A > x => d + B > Ax => d > Ax - B = Ax - nx = x(A - n) = x(4x - n - n) = x(4x - 2n)
            # So d > x(4x - 2n) = 2x(2x - n)
            
            # Let's find divisors of B^2. B = n*x, B^2 = n^2 * x^2
            # We can iterate over divisors d of B^2 where d > 2x(2x-n) (if 2x > n)
            # But B^2 can be large. Let's use the fact that if d is a divisor, so is B^2/d.
            # We want y < z, which means d < B^2/d, i.e., d < B
            
            # Actually, let's just iterate d from 1 to B (since d < B for y < z)
            # But that's too slow for large B.
            
            # Better: factorize B^2 and generate divisors, or iterate up to sqrt(B^2) = B
            # B = n*x, n up to 256, x up to ~192, so B up to ~49152, B^2 up to ~2.4e9
            # Iterating all divisors of B^2 by checking up to B is too slow.
            
            # Alternative: iterate d from 1 to B, but only check if d divides B^2
            # That's B iterations, which is up to ~50000 per x, and x ranges over ~n/2 values
            # Total: sum over n of (n/2) * n = sum n^2/2 ~ 256^3/6 ~ 2.8M * some factor
            # Actually for each n, x ranges from n/4 to 3n/4, that's n/2 values
            # For each x, B = n*x, we iterate d from 1 to B checking if B^2 % d == 0
            # That's n*x iterations per x
            # Total for one n: sum_{x=n/4}^{3n/4} n*x ~ n * (n/2) * (n/2) = n^3/8
            # For n=256: 256^3/8 = 2M iterations. For all n: sum n^3/8 ~ 256^4/32 ~ 6.5e8. Too slow.
            
            # Better approach: for each x, factorize B = n*x, generate divisors of B^2
            # Or: note that d must divide B^2, and we need (d+B) % A == 0
            # So d ≡ -B (mod A)
            # And d | B^2
            # So we need divisors d of B^2 such that d ≡ -B (mod A)
            
            # Let's try a different approach: iterate over possible y values
            # 1/y + 1/z = A/B, so y > B/A (since 1/y < A/B)
            # Also y > x
            # And z = B*y/(A*y - B) must be integer
            # So A*y - B must divide B*y
            # Let g = gcd(A, B). Then A*y - B must divide B*y
            # This is equivalent to (A*y - B) | B*y
            # Let d = A*y - B, then d | B*y = B*(d+B)/A = B*d/A + B^2/A
            # So d | B^2/A, which requires A | B^2
            # Hmm, this is getting complex.
            
            # Let's just use the divisor approach but be smarter.
            # For given x, A = 4x - n, B = n*x
            # We need d | B^2 and (d + B) % A == 0 and d < B (for y < z)
            # and y = (d+B)/A > x, z = (B^2/d + B)/A > y, all <= MAX_DENOM
            
            # Let's iterate d from 1 to B, stepping through divisors of B^2
            # We can find divisors by iterating i from 1 to sqrt(B^2) = B, but that's B steps
            # Instead, let's iterate i from 1 to B, and for each i that divides B^2,
            # check both i and B^2/i
            
            # Actually, let's just try small y values directly.
            # y must be > max(x, B/A)
            # And z = B*y / (A*y - B) must be integer and > y
            # A*y - B > 0 => y > B/A
            # z > y => B*y > y*(A*y - B) => B > A*y - B => 2B > A*y => y < 2B/A
            # So y is in range (max(x, B/A), 2B/A)
            # The range length is B/A, which is n*x/(4x-n)
            # For x close to n/4, A is small, range is large
            # For x close to 3n/4, A is large, range is small
            
            y_min = max(x + 1, B // A + 1)
            y_max = (2 * B) // A  # y < 2B/A
            
            if y_max < y_min:
                continue
            
            # Iterate y from y_min to y_max
            # For each y, check if (A*y - B) divides (B*y)
            # And z = B*y // (A*y - B) > y and <= MAX_DENOM
            # And z != x, z != y
            
            # Optimization: use the fact that A*y - B must divide B*y
            # Let g = gcd(A, B). Then A/g * y - B/g must divide (B/g)*y
            # This means A/g * y - B/g must divide (B/g)*(y - (B/g)/(A/g)) ... not quite
            # Actually, let d = A*y - B. Then d | B*y.
            # Since d = A*y - B, we have B = A*y - d.
            # So d | (A*y - d)*y = A*y^2 - d*y, so d | A*y^2.
            # Also d = A*y - B, so gcd(d, A) = gcd(A*y - B, A) = gcd(B, A) = g.
            # So d/g | (A/g)*y^2.
            # And d = A*y - B, so d/g = (A/g)*y - B/g.
            # Let A' = A/g, B' = B/g. Then d' = A'*y - B' and d' | A'*y^2.
            # Also d' = A'*y - B', so A' divides d' + B', i.e., A' | B' (mod d').
            # Hmm, this is still complex.
            
            # Let's just iterate and check.
            for y in range(y_min, y_max + 1):
                if time.time() > deadline:
                    break
                
                denom = A * y - B
                if denom <= 0:
                    continue
                
                if (B * y) % denom != 0:
                    continue
                
                z = (B * y) // denom
                
                if z <= y:
                    continue
                
                if z > MAX_DENOM:
                    continue
                
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                
                # Verify the equation
                # 1/x + 1/y + 1/z = 4/n
                # Check: n*(y*z + x*z + x*y) == 4*x*y*z
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
