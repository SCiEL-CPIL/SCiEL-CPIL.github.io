import time

DENOM_BOUND = 10**12

def construct():
    deadline = time.time() + 80.0
    answers = [None] * 254
    
    for i, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
            
        found = None
        nx = n
        
        # x must be > n/4 and <= 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            # Denominator for y bound: 4x - n
            A = 4 * x - n
            if A <= 0:
                continue
                
            N = nx * x
            
            # y > N / A
            y_lower = N // A + 1
            if y_lower < x:
                y_lower = x
                
            # y <= 2N / A
            y_upper = (2 * N) // A
            
            if y_lower > y_upper:
                continue
            
            for y in range(y_lower, y_upper + 1):
                D = A * y - N
                if D <= 0:
                    continue
                    
                if (N * y) % D != 0:
                    continue
                    
                z = (N * y) // D
                
                # We assumed x <= y <= z. Check z >= y.
                if z < y:
                    continue
                    
                # Check distinctness
                if x == y or y == z:
                    continue
                    
                # Check bound
                if z > DENOM_BOUND:
                    continue
                
                # Valid solution
                found = [x, y, z]
                break
            
            if found:
                break
                
        answers[i] = found
        
    return answers
