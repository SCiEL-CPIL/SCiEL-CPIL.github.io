import time

def construct():
    answers = []
    deadline = time.time() + 50  # Safety margin before 60s limit
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must satisfy: 1/x < 4/n (so x > n/4) and 3/x > 4/n (so x < 3n/4)
        # Also x must be positive integer
        start_x = n // 4 + 1
        # For distinctness, we need at least 3 different values, so x < 3n/4 roughly
        end_x = min(3 * n // 4, 10**12)
        
        for x in range(start_x, end_x + 1):
            if time.time() > deadline:
                break
            
            a = 4 * x - n  # numerator of (4/n - 1/x)
            b = n * x      # denominator of (4/n - 1/x)
            
            if a <= 0:
                continue
            
            # Need 1/y + 1/z = a/b
            # y must be > b/a (so that a*y - b > 0 for z to be positive)
            # Also y > x for distinctness (wlog assume x < y < z)
            # z = b*y / (a*y - b)
            # z must be integer, z > y, z != x, z != y
            
            # Lower bound for y: max(x+1, floor(b/a) + 1)
            y_min = max(x + 1, b // a + 1)
            # Upper bound: as y increases, z decreases. We need z > y.
            # z = b*y/(a*y-b) > y  =>  b*y > a*y² - b*y  =>  2b*y > a*y²  =>  y < 2b/a
            y_max = min(2 * b // a, 10**12)
            
            if y_min > y_max:
                continue
            
            for y in range(y_min, y_max + 1):
                if time.time() > deadline:
                    break
                
                denom = a * y - b
                if denom <= 0:
                    continue
                
                if (b * y) % denom != 0:
                    continue
                
                z = (b * y) // denom
                
                if z <= y:
                    continue
                if z > 10**12:
                    continue
                
                # Check distinctness
                if x != y and x != z and y != z:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
