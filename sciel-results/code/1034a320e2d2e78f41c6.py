import math
import time

def construct():
    deadline = time.time() + 85  # Safety margin before 90s limit
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must satisfy 1/x < 4/n, so x > n/4
        # Also x must be such that A = 4x - n > 0
        start_x = n // 4 + 1
        
        # Try increasing x values; usually small x works
        for x in range(start_x, start_x + 200):
            if time.time() > deadline:
                break
            
            A = 4 * x - n
            if A <= 0:
                continue
            
            B = n * x
            B2 = B * B
            
            # Find all divisors of B2
            # We need d such that d > B and B2/d > B (i.e., d < B2/B = B)
            # Wait: (A*y - B)(A*z - B) = B^2
            # Let d = A*y - B, then A*z - B = B2/d
            # y = (d + B) / A, z = (B2/d + B) / A
            # We need d > 0, B2/d > 0, so d > 0
            # y > 0 requires d + B > 0, which is true for d > 0
            # z > 0 requires B2/d + B > 0, true
            # y and z must be integers: (d + B) % A == 0 and (B2//d + B) % A == 0
            # Also y != z, y != x, z != x, y > 0, z > 0, y <= 10^12, z <= 10^12
            # And we want distinct denominators, so x, y, z all distinct
            
            # Enumerate divisors of B2
            # B2 can be up to (256 * 256)^2 = 4294967296, which is ~4.3e9
            # That's too large to enumerate all divisors naively
            # But we only need to try a limited range of x, and for each x, 
            # we can find divisors of B2 efficiently
            
            # For small B2, enumerate divisors by trial division up to sqrt(B2)
            sqrt_B2 = int(math.isqrt(B2))
            
            for d in range(1, sqrt_B2 + 1):
                if time.time() > deadline:
                    break
                
                if B2 % d != 0:
                    continue
                
                # d and B2//d are both divisors
                for dd in (d, B2 // d):
                    # y = (dd + B) / A
                    if (dd + B) % A != 0:
                        continue
                    y = (dd + B) // A
                    
                    # z = (B2//dd + B) / A
                    other = B2 // dd
                    if (other + B) % A != 0:
                        continue
                    z = (other + B) // A
                    
                    if y <= 0 or z <= 0:
                        continue
                    if y > 10**12 or z > 10**12:
                        continue
                    
                    # Check distinctness
                    if x == y or x == z or y == z:
                        continue
                    
                    # Verify the solution
                    # 1/x + 1/y + 1/z == 4/n
                    # Cross multiply: n*y*z + n*x*z + n*x*y == 4*x*y*z
                    lhs = n * y * z + n * x * z + n * x * y
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        found = [x, y, z]
                        break
                
                if found:
                    break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
