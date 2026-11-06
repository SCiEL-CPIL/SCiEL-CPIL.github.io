"""Optimized constructor for distinct three-term unit-fraction reps of 4/n."""
import time

DENOM_BOUND = 10**12

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin under 90s kill
    answers = [None] * 254  # index 0 -> n=3, ..., index 253 -> n=256
    
    for i, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        
        # x must be in (n/4, 3n/4) for x < y < z
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        found = None
        
        for x in range(x_min, x_max + 1):
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # y must satisfy: B/A < y <= 2B/A
            y_lo = B // A + 1
            y_hi = (2 * B) // A
            
            if y_lo > y_hi:
                continue
            
            # Ensure y > x for distinctness and ordering x < y < z
            if y_lo <= x:
                y_lo = x + 1
            if y_lo > y_hi:
                continue
                
            # Iterate y
            # We can break early if we find a solution
            for y in range(y_lo, y_hi + 1):
                D = A * y - B
                # D > 0 is guaranteed by y > B/A
                if (B * y) % D == 0:
                    z = (B * y) // D
                    # Check distinctness and bounds
                    if z > y and z <= DENOM_BOUND:
                        # x < y < z ensures distinctness
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        answers[i] = found
        
    return answers
