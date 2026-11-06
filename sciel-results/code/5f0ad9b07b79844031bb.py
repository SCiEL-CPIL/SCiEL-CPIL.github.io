def construct():
    answers = []
    MAX_N = 256
    MAX_DENOM = 10**12
    
    for n in range(3, MAX_N + 1):
        found = None
        start_x = (n // 4) + 1
        
        # Iterate x for a reasonable range
        # Solutions typically exist for small x.
        # We iterate x up to start_x + 1000.
        for x in range(start_x, start_x + 1000):
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # We need 1/y + 1/z = A/B
            # With y < z, we have B/A < y < 2B/A
            # y_min = floor(B/A) + 1
            # y_max = floor(2B/A)
            
            y_min = B // A + 1
            y_max = (2 * B) // A
            
            # Ensure y_min is not too small (y must be > x for distinctness if we want to avoid permutations, 
            # but actually we just need distinct x,y,z. 
            # We can assume x < y < z to simplify search and ensure distinctness.
            # If x < y < z, then y > x.
            if y_min <= x:
                y_min = x + 1
            
            # If the range is huge, we might want to limit it, but for small x, 
            # B/A is roughly n/4, which is small.
            # For x close to n/4, A is small, B/A is large.
            # However, we break as soon as we find a solution.
            
            for y in range(y_min, y_max + 1):
                # z = By / (Ay - B)
                denom = A * y - B
                if denom <= 0:
                    continue
                
                if (B * y) % denom != 0:
                    continue
                
                z = (B * y) // denom
                
                # Check distinctness and bounds
                if z <= y:
                    continue
                if x == y or x == z or y == z:
                    continue
                if x > MAX_DENOM or y > MAX_DENOM or z > MAX_DENOM:
                    continue
                
                # Verify equation: 1/x + 1/y + 1/z = 4/n
                # n(yz + xz + xy) = 4xyz
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found is not None:
                break
        
        answers.append(found)
    
    return answers
