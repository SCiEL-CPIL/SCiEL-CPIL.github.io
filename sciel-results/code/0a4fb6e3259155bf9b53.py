import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55 seconds safety margin
    
    answers = []
    MAX_DENOM = 10**12
    
    for n in range(3, 257):
        if time.time() > deadline:
            # If we hit the deadline, fill remaining with None to maintain structure
            # But given the speed, this shouldn't happen.
            answers.append(None)
            continue
            
        found = None
        # x must satisfy: 1/x < 4/n => x > n/4
        # and 3/x > 4/n => x < 3n/4
        # So x is in (n/4, 3n/4)
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            A = 4 * x - n
            B = n * x
            # We need 1/y + 1/z = A/B
            # y must be > x
            # y must satisfy: 1/y < A/B => y > B/A
            # and 2/y > A/B => y < 2B/A
            # So y is in (B/A, 2B/A)
            y_min = B // A + 1
            if y_min <= x:
                y_min = x + 1
            y_max = (2 * B) // A
            
            # Optimization: If y_max < y_min, skip
            if y_max < y_min:
                continue
                
            # Iterate y
            # We can add a small check for time inside the inner loop if n is large, 
            # but n<=256 is very fast.
            for y in range(y_min, y_max + 1):
                # z = B*y / (A*y - B)
                denom = A * y - B
                if denom <= 0:
                    continue
                num = B * y
                if num % denom == 0:
                    z = num // denom
                    # Check distinctness and bounds
                    if z > y and z <= MAX_DENOM:
                        found = [x, y, z]
                        break
            if found:
                break
                
        answers.append(found)
        
    return answers
