import time
import math

def construct():
    answers = []
    deadline = time.time() + 55  # safety margin before 60s limit
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x bounds:
        # 1/x < 4/n => x > n/4
        # 3/x > 4/n => x < 3n/4 (since 1/x+1/y+1/z < 3/x if y,z > x, but we don't assume order yet.
        # However, if x is the smallest, 1/x is the largest term.
        # Actually, just use the general bound:
        # 1/x must be less than 4/n, so x > n/4.
        # 1/x + 1/y + 1/z = 4/n. Since y,z >= 1, 4/n - 1/x <= 1/1 + 1/1 = 2? No.
        # Standard bound: if x <= y <= z, then 3/x >= 4/n => x <= 3n/4.
        # We can assume w.l.o.g. x <= y <= z to reduce search space?
        # The problem asks for ANY distinct x,y,z.
        # If we find a solution with x > y, we can swap them.
        # So we can restrict our search to x <= y <= z?
        # If we find x, y, z distinct, we can sort them.
        # Let's try to find x such that there exist y, z.
        # If we assume x <= y <= z, then 1/x >= 1/y >= 1/z.
        # 3/x >= 4/n => x <= 3n/4.
        # 1/x < 4/n => x > n/4.
        # So x in [floor(n/4)+1, floor(3n/4)] is a safe range for the smallest denominator.
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            if found:
                break
            
            # 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
            a = 4 * x - n
            b = n * x
            
            if a <= 0:
                continue
            
            # (ay - b)(az - b) = b^2
            # Let u = ay - b, v = az - b.
            # u * v = b^2
            # y = (u + b) / a
            # z = (v + b) / a
            
            # We need y, z to be integers, so (u+b) % a == 0 and (v+b) % a == 0.
            # Also y, z > 0 => u > -b, v > -b. Since u,v are divisors of b^2 (positive), this is always true.
            # Distinctness: x != y, x != z, y != z.
            # If we assume x <= y <= z, then y >= x and z >= y.
            # y >= x => (u+b)/a >= x => u+b >= ax => u >= ax - b.
            # z >= y => v >= u.
            
            b_sq = b * b
            sqrt_b_sq = math.isqrt(b_sq) # which is b
            
            # We iterate u from 1 to b.
            # If u divides b_sq, then v = b_sq // u.
            # We check the pair (u, v).
            # We can assume u <= v, which means u <= b.
            
            u_min = max(1, a * x - b)
            
            # Optimization: If u_min > b, no solution for this x with x <= y <= z.
            if u_min > b:
                continue
                
            # Iterate divisors u of b_sq in range [u_min, b]
            # Instead of finding all divisors first, we can just iterate i and check divisibility.
            # But b can be ~50000. Iterating 50000 times per x is ~50k ops.
            # Number of x values is ~ n/2 ~ 128.
            # Total ops ~ 128 * 50k = 6.4M. This is very fast in Python (0.1s).
            
            for u in range(u_min, b + 1):
                if time.time() > deadline:
                    break
                if b_sq % u != 0:
                    continue
                
                v = b_sq // u
                
                # Check divisibility by a
                if (u + b) % a != 0:
                    continue
                if (v + b) % a != 0:
                    continue
                
                y = (u + b) // a
                z = (v + b) // a
                
                # Check bounds
                if y > 10**12 or z > 10**12:
                    continue
                
                # Check ordering x <= y <= z (implied by our derivation)
                if y < x or z < y:
                    continue
                
                # Check distinctness
                # We assumed x <= y <= z.
                # Distinct means x < y < z.
                if x == y or y == z:
                    continue
                
                # Verify solution
                # 1/x + 1/y + 1/z == 4/n
                # n(yz + xz + xy) == 4xyz
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
