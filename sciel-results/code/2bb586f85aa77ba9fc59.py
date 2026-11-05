import math

def construct():
    answers = []
    for n in range(3, 257):
        found = None
        # x <= y <= z
        # 1/x >= 1/y >= 1/z
        # 3/x >= 4/n => x <= 3n/4
        # 1/x < 4/n => x > n/4
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        
        for x in range(start_x, end_x + 1):
            # Denominator for z: D = 4xy - nx - ny
            # z = nxy / D
            # We need D > 0 and D | nxy
            # Also z >= y => nxy >= y(4xy - nx - ny) => nx >= 4xy - nx - ny => 2nx + ny >= 4xy => y(4x - n) <= 2nx
            # If 4x - n <= 0, then x <= n/4, but x > n/4, so 4x - n > 0.
            # y <= 2nx / (4x - n)
            denom_y = 4 * x - n
            if denom_y <= 0:
                continue
            max_y = (2 * n * x) // denom_y
            # Also y >= x
            # And we want distinct x, y, z. So y != x and z != x and z != y.
            # Since we assume x <= y <= z, we just need x != y and y != z.
            # If x == y, skip.
            # If y == z, skip.
            
            # Optimization: if max_y < x, no solution for this x
            if max_y < x:
                continue
            
            # We can limit max_y to avoid huge loops if necessary, but the bound is tight.
            # For x close to n/4, max_y can be large.
            # Example n=256, x=65. 4x-n = 260-256=4. max_y = 2*256*65/4 = 8320.
            # 8320 iterations is fine.
            
            for y in range(x, max_y + 1):
                if x == y:
                    continue
                
                D = 4 * x * y - n * x - n * y
                if D <= 0:
                    continue
                
                numerator = n * x * y
                if numerator % D != 0:
                    continue
                
                z = numerator // D
                
                if z < y:
                    continue
                
                if y == z:
                    continue
                
                if z <= 1000000000000 and y <= 1000000000000 and x <= 1000000000000:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
    return answers
