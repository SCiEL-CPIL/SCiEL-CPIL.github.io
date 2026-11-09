import time
from math import isqrt

def construct():
    answers = []
    deadline = time.time() + 55  # Safety margin before 90s kill
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must satisfy: n/4 < x <= 3n/4 (since x <= y <= z and 3/x >= 4/n)
        start_x = n // 4 + 1
        end_x = 3 * n // 4  # x < 3n/4 strictly? Let's use <= for safety
        
        for x in range(start_x, end_x + 1):
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # We need 1/y + 1/z = a/b, i.e., a(y+z) = b, wait no:
            # 1/y + 1/z = a/b => b(y+z) = a*y*z => a*y*z - b*y - b*z = 0
            # Multiply by a: a²*y*z - a*b*y - a*b*z = 0
            # Add b²: (a*y - b)(a*z - b) = b²
            # Let d1 = a*y - b, d2 = a*z - b, then d1*d2 = b²
            # y = (d1 + b)/a, z = (d2 + b)/a
            # Need d1 | b², d2 = b²/d1, and (d1+b) % a == 0, (d2+b) % a == 0
            # Also y, z positive integers, distinct from each other and from x
            
            b2 = b * b
            sqrt_b2 = isqrt(b2)
            
            # Iterate over divisors d1 of b² where d1 <= sqrt(b2)
            # For each such d1, d2 = b2 // d1
            # Check if (d1 + b) % a == 0 and (d2 + b) % a == 0
            
            for d1 in range(1, sqrt_b2 + 1):
                if b2 % d1 != 0:
                    continue
                d2 = b2 // d1
                
                # y = (d1 + b) / a, z = (d2 + b) / a
                if (d1 + b) % a != 0:
                    continue
                if (d2 + b) % a != 0:
                    continue
                
                y = (d1 + b) // a
                z = (d2 + b) // a
                
                if y <= 0 or z <= 0:
                    continue
                if y > z:
                    y, z = z, y
                if y <= 0 or z <= 0:
                    continue
                
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                if z > 10**12:
                    continue
                
                # Verify the equation
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
