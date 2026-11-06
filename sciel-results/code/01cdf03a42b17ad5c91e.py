import time

def construct():
    answers = []
    deadline = time.time() + 45  # Safety margin
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        # x must satisfy 1/x < 4/n => x > n/4
        # and 3/x > 4/n => x < 3n/4
        min_x = n // 4 + 1
        max_x = (3 * n) // 4 + 1
        
        for x in range(min_x, max_x + 1):
            K = 4 * n - x
            if K <= 0:
                break
            
            # We need 1/y + 1/z = K / (n*x)
            # y < z => 1/y > 1/2 * K/(n*x) => y < 2*n*x / K
            # 1/y < K/(n*x) => y > n*x / K
            # Also y > x
            
            y_min = (n * x) // K + 1
            if y_min <= x:
                y_min = x + 1
                
            y_max = (2 * n * x) // K
            
            # If y_max < y_min, no solution for this x
            if y_max < y_min:
                continue
                
            for y in range(y_min, y_max + 1):
                # z = n*x*y / (K*y - n*x)
                D = K * y - n * x
                if D <= 0:
                    continue
                
                # Check divisibility: n*x*y % D == 0
                num = n * x * y
                if num % D != 0:
                    continue
                
                z = num // D
                
                # Ensure z > y to maintain x < y < z
                # This also ensures distinctness
                if z <= y:
                    continue
                
                # Check bounds
                if z > 10**12:
                    continue
                
                # Final verification
                # 1/x + 1/y + 1/z == 4/n
                # n(yz + xz + xy) == 4xyz
                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
