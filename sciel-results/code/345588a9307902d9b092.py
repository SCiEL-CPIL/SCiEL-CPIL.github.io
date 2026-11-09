import math

def construct():
    answers = []
    
    for n in range(3, 257):
        found = None
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        
        for x in range(start_x, end_x + 1):
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            B2 = B * B
            
            m_min = B // A + 1
            m_max = (2 * B) // A
            
            # If the range is too large, we might want to optimize, 
            # but for n<=256, it's fine.
            for m in range(m_min, m_max + 1):
                U = m * A - B
                if U <= 0:
                    continue
                if B2 % U != 0:
                    continue
                V = B2 // U
                if (V + B) % A != 0:
                    continue
                y = (U + B) // A
                z = (V + B) // A
                
                if x == y or x == z or y == z:
                    continue
                
                if y > 10**12 or z > 10**12:
                    continue
                
                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    found = [x, y, z]
                    break
            if found:
                break
        answers.append(found)
    return answers
