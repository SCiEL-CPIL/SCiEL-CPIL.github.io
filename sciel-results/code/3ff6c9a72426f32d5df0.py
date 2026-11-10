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

    def find_pair(c, d, forbidden):
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
        limit = int(math.isqrt(d2))
        
        # Values of U to skip:
        # 1. y = forbidden => U = c*forbidden - d
        # 2. z = forbidden => V = c*forbidden - d => U = d2 / (c*forbidden - d) if divisible
        # 3. y = z => U = V => U^2 = d2 => U = d. Check if y=z is integer.
        
        u_forbid_y = c * forbidden - d
        v_forbid_z = c * forbidden - d
        u_forbid_z = None
        if v_forbid_z != 0 and d2 % v_forbid_z == 0:
            u_forbid_z = d2 // v_forbid_z
            
        for u in range(1, limit + 1):
            if d2 % u != 0:
                continue
            
            # Skip if this u leads to y=forbidden
            if u == u_forbid_y:
                continue
                
            v = d2 // u
            
            # Skip if this u leads to z=forbidden
            # z = (v+d)/c. If z == forbidden, then v = c*forbidden - d.
            # So if v == v_forbid_z, skip.
            if v == v_forbid_z:
                continue
                
            # Check divisibility for y
            if (u + d) % c != 0:
                continue
            y = (u + d) // c
            
            # Check divisibility for z
            if (v + d) % c != 0:
                continue
            z = (v + d) // c
            
            if y <= 0 or z <= 0:
                continue
                
            # Check distinctness
            if y == z:
                continue
            if y == forbidden or z == forbidden:
                continue
                
            if y > LIMIT or z > LIMIT:
                continue
                
            return (y, z)
            
        return None

    for n in range(3, 257):
        if time.time() > deadline:
            break
            
        found = None
        
        # Try Strategy 2: Fix x, solve for y, z via 2-term.
        x_min = n // 4 + 1
        x_max = 3 * n // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            c = 4 * x - n
            if c <= 0:
                continue
                
            d = n * x
            
            pair = find_pair(c, d, x)
            if pair is not None:
                y, z = pair
                # find_pair already checks distinctness from x and each other
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
        
        if found:
            if not verify(n, *found):
                found = None
                
        answers[n - 3] = found

    return answers
