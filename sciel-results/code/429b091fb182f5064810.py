import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    answers = []
    MAX_Z = 10**12
    n_total = 254
    
    for idx, n in enumerate(range(3, 257)):
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        remaining_n = n_total - idx
        
        if remaining <= 0:
            answers.append(None)
            continue
        
        # Allocate time for this n
        # We want to be fast. Most n should take < 1ms.
        # Hard cases might take longer.
        time_per_n = min(remaining / remaining_n, 1.0)
        n_deadline = time.time() + time_per_n
        
        found = None
        
        # x range: n/4 < x <= 3n/4
        # We iterate x from 3n/4 down to n/4 + 1
        # This ensures we start with small y-ranges
        
        x_max = (3 * n) // 4
        x_min = n // 4 + 1
        
        # Ensure x_max >= x_min
        if x_max < x_min:
            x_max = x_min
            
        for x in range(x_max, x_min - 1, -1):
            if time.time() > n_deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
            
            B = n * x
            
            # y must satisfy: B/A < y < 2B/A
            # Also y > x
            y_min = B // A + 1
            if y_min <= x:
                y_min = x + 1
                
            y_max = (2 * B) // A
            
            if y_max < y_min:
                continue
                
            # Iterate y
            # To speed up, we can skip if range is too large and we've spent time
            # But let's just iterate. Python is fast enough for ~10k iterations.
            for y in range(y_min, y_max + 1):
                D = A * y - B
                if D <= 0:
                    continue
                
                # Check if By is divisible by D
                # z = By / D
                if (B * y) % D != 0:
                    continue
                    
                z = (B * y) // D
                
                # Check distinctness and bounds
                # We assumed x < y < z
                if z <= y:
                    continue
                if z > MAX_Z:
                    continue
                
                # Found a valid solution
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
