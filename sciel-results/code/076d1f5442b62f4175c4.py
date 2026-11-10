import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    LIMIT = 10**12
    answers = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        
        found = None
        n3 = n * n * n
        
        # Enumerate divisors of n^3
        # For each divisor a of n^3 with a = 4x - n > 0:
        #   x = (a + n) / 4 must be a positive integer
        #   Then bc = n^3 / a, where b = 4y - n > 0, c = 4z - n > 0
        #   y = (b + n) / 4, z = (c + n) / 4 must be positive integers
        #   x, y, z must be distinct and <= LIMIT
        
        # Get all divisors of n^3
        divisors = set()
        for d in range(1, isqrt(n3) + 1):
            if n3 % d == 0:
                divisors.add(d)
                divisors.add(n3 // d)
        
        # Sort divisors for deterministic processing
        sorted_divs = sorted(divisors)
        
        for a in sorted_divs:
            if time.time() > deadline:
                break
            
            # a = 4x - n, so x = (a + n) / 4
            if (a + n) % 4 != 0:
                continue
            x = (a + n) // 4
            if x <= 0 or x > LIMIT:
                continue
            
            # bc = n^3 / a
            bc = n3 // a
            
            # Now enumerate divisors b of bc where b = 4y - n > 0
            # and c = bc / b = 4z - n > 0
            # y = (b + n) / 4, z = (c + n) / 4
            # We need y, z to be positive integers, distinct from x and each other
            
            # To avoid duplicates, we can require b <= c (i.e., b <= sqrt(bc))
            # but actually we just need to find any valid triplet
            # Let's iterate over divisors b of bc
            
            sqrt_bc = isqrt(bc)
            for b in range(1, sqrt_bc + 1):
                if time.time() > deadline:
                    break
                if bc % b != 0:
                    continue
                
                c = bc // b
                
                # Check b = 4y - n
                if (b + n) % 4 != 0:
                    continue
                y = (b + n) // 4
                if y <= 0 or y > LIMIT:
                    continue
                
                # Check c = 4z - n
                if (c + n) % 4 != 0:
                    continue
                z = (c + n) // 4
                if z <= 0 or z > LIMIT:
                    continue
                
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                
                # Valid solution
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
