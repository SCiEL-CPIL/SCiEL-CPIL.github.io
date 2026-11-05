"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must satisfy: 1/x < 4/n => x > n/4
        # Also x < y < z, so 3/x > 4/n => x < 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4  # inclusive
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # We need 1/y + 1/z = a/b
            # y must satisfy: 1/y < a/b => y > b/a
            # Also y < z, so 2/y > a/b => y < 2b/a
            y_min = b // a + 1
            y_max = (2 * b) // a
            
            # To ensure distinctness x < y < z, we need y > x
            y_min = max(y_min, x + 1)
            
            for y in range(y_min, y_max + 1):
                if time.time() > deadline:
                    break
                
                # z = b*y / (a*y - b)
                denom = a * y - b
                if denom <= 0:
                    continue
                
                if (b * y) % denom == 0:
                    z = (b * y) // denom
                    if z > y and z != x:
                        found = [x, y, z]
                        break
            
            if found is not None:
                break
        
        answers.append(found)
    
    # Fill remaining with None if we ran out of time
    while len(answers) < 254:
        answers.append(None)
    
    return answers
# EVOLVE-BLOCK-END
