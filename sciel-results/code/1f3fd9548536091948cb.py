import math

def construct():
    answers = []
    # Precompute nothing? Or primes? 
    # The m-iteration approach doesn't need primes, just divisibility check U | B^2.
    # U | B^2 is equivalent to B^2 % U == 0.
    
    for n in range(3, 257):
        found = None
        # Try x values starting from ceil(n/4)
        # x must be > n/4 for A > 0
        start_x = n // 4 + 1
        if 4 * (n // 4) == n:
            start_x = n // 4 + 1
        else:
            start_x = n // 4 + 1
            
        # Actually just start from floor(n/4)+1
        start_x = n // 4 + 1
        
        for x in range(start_x, start_x + 100):
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            B2 = B * B
            
            # We look for U such that U | B^2, U > 0, U <= B (wlog U <= V)
            # U = m * A - B
            # m * A - B > 0 => m > B/A
            # m * A - B <= B => m <= 2B/A
            m_min = B // A + 1
            if (B % A) == 0:
                m_min = B // A + 1
            else:
                m_min = B // A + 1
            
            m_max = (2 * B) // A
            
            for m in range(m_min, m_max + 1):
                U = m * A - B
                if U <= 0:
                    continue
                # Check U divides B^2
                if B2 % U != 0:
                    continue
                V = B2 // U
                # Check V congruence
                if (V + B) % A != 0:
                    continue
                # Calculate y, z
                y = (U + B) // A
                z = (V + B) // A
                
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                
                # Check bounds
                if y > 10**12 or z > 10**12:
                    continue
                
                # Verify
                # 1/x + 1/y + 1/z == 4/n
                # n(yz + xz + xy) == 4xyz
                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    found = [x, y, z]
                    break
            if found:
                break
        answers.append(found)
    return answers
