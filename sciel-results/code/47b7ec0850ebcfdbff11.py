import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    answers = [None] * 254  # indices 0..253 correspond to n=3..256
    MAX_DENOM = 10**12
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        # We need 1/x + 1/y + 1/z = 4/n with x < y < z, all distinct positive integers
        # From 1/x + 1/y + 1/z = 4/n, we can fix x and solve for y, z
        # 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
        # Let a = 4x - n, b = nx
        # Then 1/y + 1/z = a/b => b(y+z) = ayz => ayz - by - bz = 0
        # => (ay - b)(az - b) = b^2
        # So we need factors of b^2
        
        # x must satisfy: 4/n - 1/x > 0 => x > n/4
        # Also x < y < z, so 1/x > 1/3 * 4/n => x < 3n/4
        # More precisely, since x < y < z, we have 3/x > 4/n => x < 3n/4
        # And 1/x < 4/n => x > n/4
        
        x_start = n // 4 + 1
        x_end = (3 * n) // 4  # x < 3n/4, so x <= floor(3n/4 - epsilon)
        if x_end < x_start:
            x_end = x_start
        
        for x in range(x_start, x_end + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # We need (ay - b)(az - b) = b^2
            # Let u = ay - b, v = az - b, then u*v = b^2
            # u > 0, v > 0, and u < v (since y < z)
            # y = (u + b) / a, z = (v + b) / a
            # Need a | (u + b) and a | (v + b)
            # Also y > x and z > y
            
            # y > x => (u + b)/a > x => u + b > ax => u > ax - b = a*x - n*x = x(4 - n)
            # Since a = 4x - n, ax = x(4x - n), so u > x(4x - n) - nx = 4x^2 - nx - nx = 4x^2 - 2nx
            # Hmm, let me just check y > x directly after computing
            
            # v = b^2 / u, need u | b^2
            # u ranges from 1 to b (since u < v and u*v = b^2, so u < b)
            # Actually u < b since u < v and u*v = b^2 => u^2 < b^2 => u < b
            
            # For efficiency, iterate over divisors of b^2 that are < b
            # But b can be large. Let's be smarter.
            
            # y = (u + b) / a must be integer, so u ≡ -b (mod a), i.e., u ≡ b (mod a) since -b ≡ b only if 2b ≡ 0 mod a... no.
            # u + b ≡ 0 (mod a) => u ≡ -b (mod a)
            
            # Let's find u in the range such that:
            # 1. u * v = b^2 where v = b^2 // u (so u | b^2)
            # 2. u < b (so u < v)
            # 3. (u + b) % a == 0
            # 4. y = (u + b) // a > x
            # 5. z = (b^2 // u + b) // a > y
            
            # The constraint u ≡ -b (mod a) and u | b^2 is key.
            # Let u = k*a - b for some k, but u > 0 so k*a > b.
            # Alternatively, iterate u over divisors of b^2 below b and check the modular condition.
            
            # b^2 can be huge. Instead, note that u must divide b^2.
            # Let's factor b and enumerate divisors.
            
            # For speed, if b is large, this could be slow. Let's try a different approach:
            # Iterate over possible y values in a bounded range.
            
            # From 1/y + 1/z = a/b, with y > x:
            # 1/y < a/b => y > b/a
            # 1/z > 0, and 1/y > a/b - 1/z. Since z > y, 1/z < 1/y, so a/b < 2/y => y < 2b/a
            # So b/a < y < 2b/a
            
            y_start = b // a + 1
            y_end = 2 * b // a
            if y_end < y_start:
                continue
            
            # Also y > x
            if y_start <= x:
                y_start = x + 1
            
            for y in range(y_start, y_end + 1):
                # 1/z = a/b - 1/y = (ay - b) / (by)
                # z = by / (ay - b)
                num = a * y - b
                if num <= 0:
                    continue
                den = b * y
                if den % num != 0:
                    continue
                z = den // num
                if z <= y:
                    continue
                if z > MAX_DENOM:
                    continue
                if x == y or y == z or x == z:
                    continue
                # Verify
                if x * y * n == 4 * y * z + 4 * x * z + 4 * x * y:
                    found = [x, y, z]
                    break
            if found:
                break
        
        answers[idx] = found
    
    return answers
