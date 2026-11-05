import time
import sys

def construct():
    answers = []
    deadline = time.time() + 50  # 50s safety margin
    
    for n in range(3, 257):
        found = None
        # x must satisfy n/4 < x < 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # We need 1/y + 1/z = A/B
            # y > x and z > y
            # 1/y < A/B => y > B/A
            # 2/y > A/B => y < 2B/A
            y_min = max(x + 1, B // A + 1)
            y_max = (2 * B) // A
            
            # If the range is too large, we might want to be careful, 
            # but B/A is roughly nx/(4x-n). 
            # For x close to n/4, this is large.
            # However, we can just iterate.
            
            for y in range(y_min, y_max + 1):
                # Check if z is integer
                # z = B*y / (A*y - B)
                denom = A * y - B
                if denom <= 0:
                    continue
                if (B * y) % denom == 0:
                    z = (B * y) // denom
                    if z > y:
                        # Check distinctness
                        if x != y and y != z and x != z:
                            found = [x, y, z]
                            break
            if found:
                break
        
        answers.append(found)
        
    return answers
