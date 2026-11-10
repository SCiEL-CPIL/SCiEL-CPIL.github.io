import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        
        # Special case: n divisible by 4
        # 4/n = 1/(n/4) + 3/(4n)
        # Need 3/(4n) = 1/y + 1/z
        # => 3yz = 4n(y+z)
        # => 3yz - 4ny - 4nz = 0
        # => (3y - 4n)(3z - 4n) = 16n^2
        if n % 4 == 0:
            x = n // 4
            # Find divisors d of 16n^2 such that 3y - 4n = d, 3z - 4n = 16n^2/d
            # y = (d + 4n)/3, z = (16n^2/d + 4n)/3
            # Need d + 4n ≡ 0 (mod 3) and 16n^2/d + 4n ≡ 0 (mod 3)
            # Also y, z must be distinct from x and from each other, and positive
            
            target = 16 * n * n
            # Enumerate divisors of target
            # target = 16 * n^2, n = 4k, so target = 16 * 16k^2 = 256k^2
            # Divisors come in pairs
            
            best = None
            # We need (d + 4n) % 3 == 0 and (target//d + 4n) % 3 == 0
            # Let's try small and large divisors
            
            # Generate divisors of target
            # target = 16 * n^2
            # For n up to 256, n^2 up to 65536, target up to 1048576
            # sqrt(target) up to 1024
            
            import math
            sqrt_t = int(math.isqrt(target))
            
            for d in range(1, sqrt_t + 1):
                if time.time() > deadline:
                    break
                if target % d != 0:
                    continue
                
                e = target // d
                
                # Try (d, e) and (e, d)
                for dd, ee in [(d, e), (e, d)]:
                    if (dd + 4 * n) % 3 != 0:
                        continue
                    if (ee + 4 * n) % 3 != 0:
                        continue
                    
                    y = (dd + 4 * n) // 3
                    z = (ee + 4 * n) // 3
                    
                    if y <= 0 or z <= 0:
                        continue
                    if y == x or z == x or y == z:
                        continue
                    if y > 10**12 or z > 10**12:
                        continue
                    
                    # Verify: 1/x + 1/y + 1/z = 4/n
                    # Cross multiply: n(yz + xz + xy) = 4xyz
                    lhs = n * (y * z + x * z + x * y)
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        best = [x, y, z]
                        break
                
                if best:
                    break
            
            if best:
                found = best
        
        if not found:
            # General search
            # 1/x + 1/y + 1/z = 4/n
            # Assume x <= y <= z
            # x must satisfy: 1/x > 4/n / 3, so x < 3n/4
            # Also 1/x < 4/n, so x > n/4
            # More precisely, for distinct positive integers:
            # x_min: 1/x + 1/x + 1/x >= 4/n => x <= 3n/4
            # x_max: 1/x < 4/n => x > n/4, so x >= n//4 + 1
            
            x_min = n // 4 + 1
            x_max = (3 * n) // 4
            
            # Try x values, prefer those that give good y,z
            # For each x: 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
            # Let a = 4x - n, b = nx
            # 1/y + 1/z = a/b
            # b(y+z) = ayz
            # (ay - b)(az - b) = b^2
            # Need a > 0, i.e., x > n/4
            
            for x in range(x_min, x_max + 1):
                if (x - x_min) % 30 == 0 and time.time() > deadline:
                    break
                
                a = 4 * x - n
                if a <= 0:
                    continue
                
                b = n * x
                
                # (ay - b)(az - b) = b^2
                # Let p = ay - b, q = az - b, pq = b^2
                # y = (p + b)/a, z = (q + b)/a
                # Need p + b ≡ 0 (mod a), q + b ≡ 0 (mod a)
                # p and q are divisors of b^2 with p <= q
                # Also y > x (since we want x <= y <= z and distinct)
                # y = (p+b)/a > x => p + b > ax => p > ax - b = a*x - n*x = x(a-n) = x(4x-n-n) = x(4x-2n)
                
                # Enumerate divisors p of b^2 where p <= b (since p*q = b^2 and p<=q)
                # b^2 can be large. For n up to 256, x up to 192, b up to 256*192 = 49152
                # b^2 up to ~2.4e9, sqrt up to ~49000
                
                import math
                b2 = b * b
                sqrt_b2 = int(math.isqrt(b2))
                
                found_inner = False
                # Try p from 1 to sqrt_b2
                # p must divide b^2
                # y = (p + b) / a must be integer and > x
                # z = (b^2/p + b) / a must be integer and > y
                
                for p in range(1, sqrt_b2 + 1):
                    if (p - 1) % 100 == 0 and time.time() > deadline:
                        break
                    if b2 % p != 0:
                        continue
                    
                    q = b2 // p
                    
                    # Check p and q
                    for pp, qq in [(p, q), (q, p)]:
                        if (pp + b) % a != 0:
                            continue
                        if (qq + b) % a != 0:
                            continue
                        
                        y = (pp + b) // a
                        z = (qq + b) // a
                        
                        if y <= 0 or z <= 0:
                            continue
                        if y <= x or z <= y:
                            continue
                        if y == x or z == x or y == z:
                            continue
                        if y > 10**12 or z > 10**12:
                            continue
                        
                        # Verify
                        lhs = n * (y * z + x * z + x * y)
                        rhs = 4 * x * y * z
                        if lhs == rhs:
                            found = [x, y, z]
                            found_inner = True
                            break
                    
                    if found_inner:
                        break
                
                if found:
                    break
        
        answers.append(found)
    
    return answers
