"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _find_solution(n, deadline):
    """Find distinct positive integers x,y,z with 1/x+1/y+1/z = 4/n, x<y<z, z<=1e12."""
    # x must satisfy: 1/x < 4/n => x > n/4
    # Also from 1/x + 1/y + 1/z = 4/n with x<y<z:
    # 3/x > 4/n => x < 3n/4
    # So x is in range (n/4, 3n/4)
    x_start = n // 4 + 1
    x_end = (3 * n) // 4  # inclusive upper bound, x < 3n/4
    
    for x in range(x_start, x_end + 1):
        if time.time() > deadline:
            return None
        # 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
        # Let A = 4x - n, B = nx
        # 1/y + 1/z = A/B
        # (y+z)/(yz) = A/B => B(y+z) = A*y*z
        # A*y*z - B*y - B*z = 0
        # (A*y - B)(A*z - B) = B^2
        A = 4 * x - n
        B = n * x
        if A <= 0:
            continue
        
        # We need to find divisors d of B^2 such that:
        # d = A*y - B and B^2/d = A*z - B
        # y = (d + B) / A, z = (B^2/d + B) / A
        # Need y > x, z > y, all integers
        
        B2 = B * B
        
        # y must be > x, so A*y - B > A*x - B = A*(n//4+1) - B... 
        # Actually y > x means d = A*y - B > A*x - B
        # A*x - B = A*x - n*x = x*(A-n) = x*(4x-2n) = 2x*(2x-n)
        # Let's just iterate over possible y values more directly for small ranges
        
        # Alternative: iterate y from x+1 upward
        # 1/z = A/B - 1/y = (A*y - B) / (B*y)
        # z = B*y / (A*y - B)
        # Need A*y - B > 0 => y > B/A = n*x/(4x-n)
        # Also z > y => B*y / (A*y - B) > y => B > A*y - B => A*y < 2B => y < 2B/A
        
        y_min = B // A + 1
        if y_min <= x:
            y_min = x + 1
        y_max = (2 * B) // A  # y < 2B/A, so y_max = floor(2B/A)
        
        # Limit y search for speed
        # For large n, the range might be big. Let's cap it.
        max_y_iter = 200000
        if y_max - y_min + 1 > max_y_iter:
            y_max = y_min + max_y_iter - 1
        
        for y in range(y_min, y_max + 1):
            if time.time() > deadline:
                return None
            denom = A * y - B
            if denom <= 0:
                continue
            # z = B*y / denom
            if (B * y) % denom != 0:
                continue
            z = (B * y) // denom
            if z > y:
                # Verify
                # 1/x + 1/y + 1/z == 4/n
                # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs and x != y and y != z and x != z:
                    if z <= 1000000000000:
                        return [x, y, z]
    return None

def construct():
    answers = [None] * 254
    deadline = time.time() + 85.0  # 5 second safety margin
    
    # Precomputed known solutions for difficult cases (if any)
    # Let's just search with the improved algorithm
    
    for i, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        sol = _find_solution(n, deadline)
        if sol is not None:
            answers[i] = sol
    
    return answers
# EVOLVE-BLOCK-END
