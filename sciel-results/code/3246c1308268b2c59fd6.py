"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
            
            # (Ay-B)(Az-B) = B^2
            # Let d = Ay-B, e = Az-B, then d*e = B^2
            # y = (d+B)/A, z = (e+B)/A
            B_sq = B * B
            sqrt_B_sq = int(math.isqrt(B_sq))
            
            # d must satisfy: d >= Ax - B (so y >= x), d <= sqrt(B_sq) (so d <= e, i.e., y <= z)
            d_min = max(1, A * x - B)
            d_max = sqrt_B_sq
            
            if d_min > d_max:
                continue
            
            found_for_x = None
            for d in range(d_min, d_max + 1):
                if time.time() > deadline:
                    break
                if B_sq % d != 0:
                    continue
                e = B_sq // d
                if (d + B) % A != 0:
                    continue
                y = (d + B) // A
                if (e + B) % A != 0:
                    continue
                z = (e + B) // A
                
                if y < x or z < y:
                    continue
                if x == y or y == z or x == z:
                    continue
                if z > 10**12:
                    continue
                
                found_for_x = [x, y, z]
                break
            
            if found_for_x:
                found = found_for_x
                break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
