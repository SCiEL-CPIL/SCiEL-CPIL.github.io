import time
import math

def construct():
    deadline = time.time() + 75.0
    answers = [None] * 254
    BOUND = 10**12
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None
        
        # x must satisfy 4x > n (so 4x - n > 0), and x >= 1
        # Also for distinct positive denominators, x, y, z all >= 1
        # 1/x <= 4/n means x >= n/4, but we need 4x - n > 0 so x > n/4
        # Minimum x: floor(n/4) + 1
        x_min = n // 4 + 1
        
        # Upper bound on x: since 1/x + 1/y + 1/z = 4/n and y, z > x (WLOG x < y < z),
        # we have 4/n > 3/x, so x > 3n/4. Actually 4/n >= 3/x (since y,z >= x+1 > x),
        # but more carefully: 1/x + 1/y + 1/z < 3/x (since y,z > x), so 4/n < 3/x, x < 3n/4.
        # Wait: 1/x + 1/y + 1/z = 4/n, and if x < y < z then 1/x > 1/y > 1/z,
        # so 4/n < 3/x → x < 3n/4. Also 4/n > 1/x → x > n/4.
        # So x is in (n/4, 3n/4).
        x_max = (3 * n) // 4  # x < 3n/4, so x <= ceil(3n/4) - 1, but let's use a safe upper bound
        
        # Actually, let's be more careful. We want x < y < z (distinct, sorted).
        # 4/n = 1/x + 1/y + 1/z < 1/x + 1/x + 1/x = 3/x, so x < 3n/4.
        # 4/n = 1/x + 1/y + 1/z > 1/x, so x > n/4.
        # So x ranges from floor(n/4)+1 to ceil(3n/4)-1.
        
        x_start = n // 4 + 1
        x_end = (3 * n + 3) // 4  # safe upper bound
        
        for x in range(x_start, x_end + 1):
            if time.time() > deadline:
                break
            if found:
                break
            
            a = 4 * x - n  # positive since x > n/4
            b = n * x      # positive
            
            # We need (ay - b)(az - b) = b^2
            # Let u = ay - b, v = az - b, so u*v = b^2
            # y = (u + b) / a, z = (v + b) / a
            # Need: u, v positive, u*v = b^2, (u+b) % a == 0, (v+b) % a == 0
            # y > x, z > y, all distinct, all <= BOUND
            
            b_sq = b * b
            
            # Find divisors u of b^2 such that u < b^2 (so v > 1) and u > 0
            # We need u ≡ -b (mod a) and v ≡ -b (mod a) where v = b^2/u
            # Since u*v = b^2, v = b^2/u must be integer.
            
            # Enumerate divisors of b^2. b^2 can be large, but b = n*x <= 256*192 ≈ 49152,
            # so b^2 <= ~2.4e9. Number of divisors is manageable.
            
            # Find all divisors of b_sq
            # For efficiency, iterate u up to sqrt(b_sq)
            limit = math.isqrt(b_sq)
            
            found_pair = False
            for u in range(1, limit + 1):
                if b_sq % u != 0:
                    continue
                v = b_sq // u
                # Check congruence: (u + b) % a == 0 and (v + b) % a == 0
                if (u + b) % a != 0:
                    continue
                if (v + b) % a != 0:
                    continue
                
                y = (u + b) // a
                z = (v + b) // a
                
                # Ensure y and z are positive integers (they are by construction)
                # Need x, y, z all distinct, positive, <= BOUND
                # And we want them to be a valid solution
                if y <= 0 or z <= 0:
                    continue
                if y > BOUND or z > BOUND or x > BOUND:
                    continue
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                # Check the equation
                # 1/x + 1/y + 1/z should equal 4/n
                # Cross multiply: n(yz + xz + xy) = 4xyz
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    # Valid solution
                    found = [x, y, z]
                    found_pair = True
                    break
            
            if found_pair:
                break
    
        answers[idx] = found
    
    return answers
