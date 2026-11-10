import time

def construct():
    answers = []
    deadline = time.time() + 85.0  # 5s safety margin
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        # x must be > n/4
        # Also 3/x > 4/n => x < 3n/4 is not strictly required for existence, 
        # but if x is too large, 1/x is small, requiring 1/y+1/z to be close to 4/n.
        # Actually, if x is the smallest denominator, 3/x > 4/n => x < 3n/4.
        # However, we don't enforce x is smallest, just distinct.
        # But iterating x in [n//4 + 1, ...] is a good start.
        
        x_start = n // 4 + 1
        # We can iterate x up to a reasonable bound. 
        # If x is very large, y and z must be close to each other.
        # Let's try x up to 10*n or until found.
        # For n=256, x_start ~ 64.
        
        for x in range(x_start, x_start + 1000): # Cap x iterations to avoid hanging on hard cases
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # y > B/A and y < 2B/A
            # y_min is smallest integer > B/A
            y_min = B // A + 1
            # y_max is largest integer < 2B/A
            y_max = (2 * B - 1) // A 
            
            if y_max < y_min:
                continue
                
            # We also need y > x to ensure distinctness easily? 
            # Not strictly, but if we find y <= x, we can swap later or just reject 
            # if we enforce x < y < z. 
            # Let's enforce x < y < z to simplify distinctness check.
            if y_min <= x:
                y_min = x + 1
                
            for y in range(y_min, y_max + 1):
                denom = A * y - B
                # denom must be > 0 (guaranteed by y > B/A)
                # z = B * y / denom
                # Check divisibility
                if (B * y) % denom == 0:
                    z = (B * y) // denom
                    if z > y and z <= 10**12:
                        found = [x, y, z]
                        break
            if found:
                break
                
        answers.append(found)
        
    return answers
