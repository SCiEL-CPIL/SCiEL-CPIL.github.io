"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    answers = []
    LIMIT = 10**12
    
    for n in range(3, 257):
        found = None
        # x must satisfy 1/x < 4/n => x > n/4
        # and 3/x > 4/n => x < 3n/4
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        
        for x in range(start_x, end_x + 1):
            A = 4 * x - n
            B = n * x
            
            # We derived:
            # 1/y + 1/z = A/B
            # Let K = A*y - B. Then z = B*y/K.
            # K must divide B*y.
            # Let d = B/K, so K = B/d. d must be a divisor of B.
            # y = (K+B)/A = (B/d + B)/A = B(d+1)/(A*d).
            # So A*d must divide B*(d+1).
            
            # Iterate over divisors d of B
            # To ensure x < y < z, we can just check at the end.
            # y = B(d+1)/(A*d).
            # z = B*y/K = B*y/(B/d) = d*y.
            # So z = d * y.
            # Condition x < y < z implies x < y and y < d*y => 1 < d.
            # So d must be > 1.
            
            # Get divisors of B
            # B is small (<= 50000), so finding divisors is fast.
            divisors = []
            i = 1
            while i * i <= B:
                if B % i == 0:
                    divisors.append(i)
                    if i != B // i:
                        divisors.append(B // i)
                i += 1
            
            for d in divisors:
                if d <= 1:
                    continue
                # Check if A*d divides B*(d+1)
                num = B * (d + 1)
                den = A * d
                if num % den == 0:
                    y = num // den
                    z = d * y
                    # Check distinctness and bounds
                    if x < y < z and z <= LIMIT:
                        found = [x, y, z]
                        break
            if found:
                break
        answers.append(found)
    return answers
# EVOLVE-BLOCK-END
