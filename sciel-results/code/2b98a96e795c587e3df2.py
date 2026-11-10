"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def _divisors(n):
    """Return all positive divisors of n."""
    divs = set()
    limit = int(math.isqrt(n))
    for i in range(1, limit + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return divs

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if (x - x_min) % 50 == 0 and time.time() > deadline:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            
            b = n * x
            b_sq = b * b
            
            # From (ay-b)(az-b) = b^2, let D = ay-b
            # y = (D+b)/a, z = (b_sq/D + b)/a
            divs = _divisors(b_sq)
            
            for D in divs:
                if (D + b) % a != 0:
                    continue
                y = (D + b) // a
                if y <= 0 or y > 10**12:
                    continue
                num_z = b_sq // D + b
                if num_z % a != 0:
                    continue
                z = num_z // a
                if z <= 0 or z > 10**12:
                    continue
                if x == y or y == z or x == z:
                    continue
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
