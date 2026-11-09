import time

def construct():
    answers = []
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    MAX_DENOM = 10**12
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        # x must satisfy 1/x < 4/n => x > n/4
        # and 1/x >= 4/n - 1/y - 1/z. Assuming x <= y <= z, 3/x >= 4/n => x <= 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Iterate x
        for x in range(x_min, x_max + 1):
            A = 4 * x - n
            B = n * x
            
            # We need y > B/A and y <= 2B/A
            # y_min is smallest integer > B/A
            # y_max is largest integer <= 2B/A
            y_min = B // A + 1
            y_max = (2 * B) // A
            
            # Ensure y > x to maintain x < y < z ordering for distinctness
            if y_min <= x:
                y_min = x + 1
            
            if y_min > y_max:
                continue
                
            for y in range(y_min, y_max + 1):
                # z = B*y / (A*y - B)
                D = A * y - B
                # D must be positive, which is guaranteed by y > B/A
                if B * y % D == 0:
                    z = (B * y) // D
                    if z > y and z <= MAX_DENOM:
                        # Distinctness x < y < z is satisfied
                        found = [x, y, z]
                        break
            if found:
                break
        answers.append(found)
        
    return answers
