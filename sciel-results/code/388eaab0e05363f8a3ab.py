import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    LIMIT = 10**12
    answers = [None] * 254
    
    def verify(n, x, y, z):
        if x is None or y is None or z is None:
            return False
        if x <= 0 or y <= 0 or z <= 0:
            return False
        if x > LIMIT or y > LIMIT or z > LIMIT:
            return False
        if x == y or y == z or x == z:
            return False
        # 1/x + 1/y + 1/z = 4/n
        # (yz + xz + xy) * n = 4 * x * y * z
        lhs = (y*z + x*z + x*y) * n
        rhs = 4 * x * y * z
        return lhs == rhs

    # Precompute divisors for small numbers?
    # Or just use a helper.
    
    def divisors(num):
        if num == 1:
            return [1]
        ds = []
        i = 1
        while i * i <= num:
            if num % i == 0:
                ds.append(i)
                if i != num // i:
                    ds.append(num // i)
            i += 1
        return ds

    # Helper to find solution for 1/a + 1/b = c/d
    # Returns (a, b) or None
    def solve_two_term(a, b, c, d):
        # 1/a + 1/b = c/d
        # d(a+b) = abc
        # abc - da - db = 0
        # Multiply by c: c^2 ab - acd a - acd b = 0
        # Add (acd)^2: (ca - cd)(cb - cd) = (cd)^2? No.
        # Standard trick: (ca - d)(cb - d) = d^2
        # Let's verify: c^2 ab - cd a - cd b + d^2 = d^2 => c^2 ab - cd(a+b) = 0 => c ab = d(a+b).
        # So (ca - d)(cb - d) = d^2.
        # Let X = ca - d, Y = cb - d.
        # X * Y = d^2.
        # a = (X + d) / c
        # b = (Y + d) / c
        # We need a, b integers.
        # Iterate divisors X of d^2.
        # Check if (X + d) % c == 0.
        # Then a = (X + d) // c.
        # Y = d^2 // X.
        # Check if (Y + d) % c == 0.
        # Then b = (Y + d) // c.
        # Return (a, b).
        
        if c <= 0 or d <= 0:
            return None
        
        d2 = d * d
        # To speed up, we can iterate up to sqrt(d2) = d.
        # But d can be up to n*x/2 ~ 256*250 ~ 64000.
        # d^2 ~ 4e9. Iterating divisors by looping to sqrt(d^2) is O(d).
        # 64000 iterations is fine.
        
        # We can be smarter.
        # We need X | d^2 and (X + d) % c == 0.
        # X = k * c - d.
        # k * c - d must be a divisor of d^2.
        # And k * c - d > 0 => k > d/c.
        # Also X <= d^2 => k c - d <= d^2 => k <= (d^2 + d) / c.
        
        # Let's just iterate divisors of d^2 by finding prime factors?
        # No, simple loop up to sqrt(d^2) is safer and fast enough for small d.
        # But wait, d can be larger if we use other strategies.
        # In the 3-term reduction, d = n*x/2. Max n=256, x~190. d ~ 24320.
        # sqrt(d^2) = 24320. Loop is very fast.
        
        for i in range(1, int(math.isqrt(d2)) + 1):
            if d2 % i == 0:
                X = i
                Y = d2 // i
                # Case 1: X = i
                if (X + d) % c == 0:
                    a = (X + d) // c
                    if (Y + d) % c == 0:
                        b = (Y + d) // c
                        if a > 0 and b > 0:
                            return (a, b)
                # Case 2: X = Y (if square) handled above
                # What if X and Y are swapped?
                # a = (Y+d)//c, b = (X+d)//c.
                # This is just swapping a and b. Since order doesn't matter for existence,
                # and we can sort later, checking one direction is enough for finding *a* solution.
                # However, distinctness in 3 terms depends on specific values.
                # (a,b) vs (b,a) gives same set.
        
        return None

    # Strategy 1: Direct 3-term search (same as current, but maybe optimized)
    # Strategy 2: Reduction to 2-term.
    # 1/x + 1/y + 1/z = 4/n.
    # Pick x.
    # 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx).
    # Let c = 4x - n, d = nx.
    # We need 1/y + 1/z = c/d.
    # Use solve_two_term(1, 1, c, d) to find (y, z).
    
    # Note: The standard reduction 1/a + 1/b = c/d leads to (ca-d)(cb-d)=d^2.
    # Here c=1 for 1/y+1/z? No.
    # The equation is 1/y + 1/z = c/d.
    # So in the formula, the "c" coefficient of 1/y is 1.
    # So we call solve_two_term with a_coeff=1, b_coeff=1?
    # My function solve_two_term(a, b, c, d) solves 1/a + 1/b = c/d.
    # So yes, a=1, b=1 is not right. The function finds values for the denominators.
    # Let's rename parameters in my head.
    # def find_pair(c, d): finds y,z such that 1/y+1/z=c/d.
    
    def find_pair(c, d):
        # 1/y + 1/z = c/d
        # d(y+z) = cyz
        # cyz - dy - dz = 0
        # Multiply by c: c^2 yz - cdy - cdz = 0
        # (cy - d)(cz - d) = d^2
        # Let U = cy - d, V = cz - d.
        # U V = d^2.
        # y = (U+d)/c, z = (V+d)/c.
        
        if c <= 0 or d <= 0:
            return None
            
        d2 = d * d
        # Iterate divisors of d^2
        # Optimization: We can skip if d is too large?
        # Max d = n*x/2. n<=256, x<=192. d <= 24576.
        # d^2 ~ 6e8. sqrt ~ 24000.
        # Looping 24000 times per x is fast.
        
        # We can iterate up to sqrt(d2)
        limit = int(math.isqrt(d2))
        for u in range(1, limit + 1):
            if d2 % u != 0:
                continue
            v = d2 // u
            # Check u
            if (u + d) % c == 0:
                y = (u + d) // c
                if (v + d) % c == 0:
                    z = (v + d) // c
                    if y > 0 and z > 0:
                        return (y, z)
            # Check v (if u != v)
            # If u != v, then y, z are swapped.
            # (y, z) from u is (y1, z1).
            # (y, z) from v is (y2, z2) = (z1, y1).
            # Same pair.
            
        return None

    # Let's try Strategy 2 for all n.
    # If it fails, fall back to Strategy 1 (direct search) or other heuristics.
    
    for n in range(3, 257):
        if time.time() > deadline:
            break
            
        found = None
        
        # Try Strategy 2: Fix x, solve for y, z via 2-term.
        # x range: n/4 < x <= 3n/4.
        # Actually, for 1/y+1/z = c/d to have solutions, we need c/d > 0.
        # c = 4x - n.
        # If 4x - n <= 0, then 1/x >= 4/n, so 1/y+1/z <= 0, impossible for positive y,z.
        # So x > n/4.
        
        # Also, for 1/y+1/z to be achievable, max value is 2/min(y,z).
        # So c/d < 2/y_min.
        
        x_min = n // 4 + 1
        x_max = 3 * n // 4
        
        # We can try specific x values first?
        # Or iterate all.
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            c = 4 * x - n
            if c <= 0:
                continue
                
            d = n * x
            
            pair = find_pair(c, d)
            if pair is not None:
                y, z = pair
                if y != x and z != x and y != z:
                    if y > 0 and z > 0 and y <= LIMIT and z <= LIMIT:
                        # Sort to be tidy, though not required
                        found = [x, y, z]
                        break
        
        if found is None:
            # Fallback to Strategy 1 if Strategy 2 failed
            # This is the exhaustive search from the previous version.
            for x in range(x_min, x_max + 1):
                if time.time() > deadline:
                    break
                A = 4 * x - n
                if A <= 0:
                    continue
                B = n * x
                
                # y range
                # 1/y + 1/z = A/B
                # y < 2B/A, y > B/A
                # y > x
                
                if (2 * B) % A == 0:
                    y_max = (2 * B) // A - 1
                else:
                    y_max = (2 * B) // A
                
                y_min = max(x + 1, B // A + 1)
                
                if y_min > y_max:
                    continue
                
                for y in range(y_min, y_max + 1):
                    if time.time() > deadline:
                        break
                    denom = A * y - B
                    if denom <= 0:
                        continue
                    if (B * y) % denom == 0:
                        z = (B * y) // denom
                        if z > y and z <= LIMIT:
                            if x != y and y != z and x != z:
                                found = [x, y, z]
                                break
                if found:
                    break
        
        # Verify before storing
        if found:
            if not verify(n, *found):
                found = None
                
        answers[n - 3] = found

    return answers
