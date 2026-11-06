import time

def construct():
    """
    Find distinct three-term unit-fraction representations of 4/n for n=3..256.
    Returns a list of 254 rows, each None or [x, y, z] with 1/x + 1/y + 1/z = 4/n.
    """
    deadline = time.time() + 80  # Safety margin before 90s kill
    answers = [None] * 254  # indices 0..253 correspond to n=3..256
    
    for idx in range(254):
        n = idx + 3
        
        if time.time() > deadline - 1:
            break
        
        found = None
        
        # x must satisfy: n/4 < x < 3n/4 (for x <= y <= z)
        # More precisely: x > n/4 and 3/x >= 4/n => x <= 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n - 1) // 4
        
        # Try x from x_max down to x_min (larger x gives smaller y range)
        for x in range(x_max, x_min - 1, -1):
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # 1/y + 1/z = A/B
            # y must satisfy: B/A < y < 2B/A (for y <= z)
            y_min = B // A + 1
            y_max = (2 * B) // A
            
            if y_min <= x:
                y_min = x + 1
            
            if y_min > y_max:
                continue
            
            # Iterate y
            for y in range(y_min, y_max + 1):
                denom = A * y - B
                if denom <= 0:
                    continue
                num = B * y
                if num % denom != 0:
                    continue
                z = num // denom
                if z <= y:
                    continue
                # x < y < z ensures distinctness
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
