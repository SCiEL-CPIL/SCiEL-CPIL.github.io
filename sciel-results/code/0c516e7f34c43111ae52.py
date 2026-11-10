import math

def construct():
    answers = []
    for n in range(3, 257):
        found = None
        # x must satisfy 1/x < 4/n => x > n/4
        # and 3/x > 4/n => x < 3n/4 (assuming x is the smallest)
        # So x is in [floor(n/4) + 1, floor(3n/4)]
        x_start = n // 4 + 1
        x_end = (3 * n) // 4
        
        for x in range(x_start, x_end + 1):
            # 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
                
            g = math.gcd(A, B)
            A_prime = A // g
            B_prime = B // g
            
            # Equation: (A' y - B')(A' z - B') = B'^2
            # Let U = A' y - B', V = A' z - B'
            # U V = B'^2
            # y = (U + B') / A'
            # z = (V + B') / A'
            
            B_sq = B_prime * B_prime
            sqrt_B_sq = int(math.isqrt(B_sq))
            
            # Iterate divisors U of B_sq
            for U in range(1, sqrt_B_sq + 1):
                if B_sq % U != 0:
                    continue
                
                V = B_sq // U
                if U >= V:
                    continue
                    
                # Check if y is integer
                if (U + B_prime) % A_prime != 0:
                    continue
                
                y = (U + B_prime) // A_prime
                
                # Check if z is integer
                if (V + B_prime) % A_prime != 0:
                    continue
                
                z = (V + B_prime) // A_prime
                
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                
                # Check bounds
                if y > 1000000000000 or z > 1000000000000:
                    continue
                
                # We found a solution
                found = [x, y, z]
                break
            
            if found:
                break
                
        answers.append(found)
    return answers
