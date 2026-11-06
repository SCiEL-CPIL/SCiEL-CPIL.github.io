import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin well under 60s
    
    answers = [None] * 254
    LIMIT = 10**12
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
            
        found = None
        
        # x is the smallest denominator
        # 1/x < 4/n => x > n/4
        # 3/x > 4/n => x < 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            # 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
            num_yz = 4 * x - n
            if num_yz <= 0:
                continue
                
            den_yz = n * x
            
            # 1/y < (4x-n)/(nx) => y > nx/(4x-n)
            # 1/y + 1/z = num_yz/den_yz
            # Since y < z, 2/y > num_yz/den_yz => y < 2*den_yz/num_yz
            y_min = max(x + 1, den_yz // num_yz + 1)
            y_max = (2 * den_yz) // num_yz
            
            # Ensure y_max is not too large, though theoretically bounded
            # Also y must be <= LIMIT
            if y_max > LIMIT:
                y_max = LIMIT
                
            for y in range(y_min, y_max + 1):
                # 1/z = (4x - n)/(nx) - 1/y
                # z = nxy / ((4x - n)y - nx)
                D = num_yz * y - den_yz
                if D <= 0:
                    continue
                
                N = den_yz * y
                if N % D == 0:
                    z = N // D
                    # We need distinct x, y, z.
                    # We enforced x < y.
                    # We need z > y to ensure z != y and z != x.
                    if z > y and z <= LIMIT:
                        found = [x, y, z]
                        break
            
            if found:
                break
                
        answers[idx] = found
        
    return answers
