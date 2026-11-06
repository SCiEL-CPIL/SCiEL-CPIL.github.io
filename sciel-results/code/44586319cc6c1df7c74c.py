def construct():
    answers = []
    for n in range(3, 257):
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Iterate x
        for x in range(x_min, x_max + 1):
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # We need u * v = B^2
            # u = kA - B
            # v = B^2 / u
            # Conditions:
            # 1. u > 0 => kA > B => k > B/A
            # 2. u divides B^2
            # 3. v = B^2 / u is integer
            # 4. v = m A - B for some integer m? 
            #    Actually, we just need y = (u+B)/A to be integer.
            #    This requires u+B to be divisible by A.
            #    u = kA - B => u+B = kA. This is always divisible by A.
            #    So y is always an integer if u is of the form kA-B.
            #    Similarly, z = (v+B)/A must be integer.
            #    This requires v+B to be divisible by A.
            #    v = B^2/u.
            #    So we need (B^2/u + B) % A == 0.
            
            k_start = B // A + 1
            # Limit k to avoid infinite loops if no solution for this x
            # B^2 is at most (256*192)^2 ~ 2.4 * 10^9.
            # A is at least 1.
            # Max k is approx B^2/A.
            # We can just iterate k while u <= B^2
            for k in range(k_start, (B * B) // A + 2):
                u = k * A - B
                if u <= 0:
                    continue
                if (B * B) % u != 0:
                    continue
                v = (B * B) // u
                # Check if z is integer
                # z = (v + B) / A
                if (v + B) % A != 0:
                    continue
                y = (u + B) // A
                z = (v + B) // A
                
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                # Check positive
                if y > 0 and z > 0:
                    found = [x, y, z]
                    break
            if found:
                break
        
        answers.append(found)
    return answers
