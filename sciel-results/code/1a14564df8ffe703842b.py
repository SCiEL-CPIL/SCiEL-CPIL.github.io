import time

def construct():
    answers = []
    deadline = time.time() + 50.0  # Safety margin
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must be at least ceil(n/4)
        x_min = (n + 3) // 4
        # x must be at most floor(3n/4) for x <= y <= z to be possible
        # 3/x >= 4/n => x <= 3n/4
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            A = 4 * x - n
            B = n * x
            
            # We need U * V = B^2
            # y = (U + B) / A, z = (V + B) / A
            # Conditions: A | (U+B) and A | (V+B)
            # U = k*A - B mod A => U ≡ -B (mod A)
            
            r = (-B) % A
            if r == 0:
                r = A
            
            # Iterate U in steps of A
            # U must be <= B (since V >= U for y <= z, and we can assume U <= V)
            # Actually, if U > V, then y > z. We just need distinctness.
            # But iterating U up to B covers all divisors pairs {U, V} with U <= V.
            # If a solution has U > V, it will be found when we pick the other divisor?
            # No, if we iterate U up to B, we cover all divisors U <= B.
            # V = B^2 / U >= B.
            # So we cover all pairs where U <= V.
            # This is sufficient because the equation is symmetric in y, z.
            
            for U in range(r, B + 1, A):
                if (B * B) % U != 0:
                    continue
                
                V = (B * B) // U
                
                # Check if V + B is divisible by A
                if (V + B) % A != 0:
                    continue
                
                y = (U + B) // A
                z = (V + B) // A
                
                # Check bounds
                if y > 10**12 or z > 10**12:
                    continue
                
                # Check distinctness
                # x, y, z must be distinct
                if x == y or x == z or y == z:
                    continue
                
                # Also ensure positive integers (implied by construction if A>0, B>0, U>0)
                # A = 4x - n. Since x >= ceil(n/4), A >= 1.
                # B = nx > 0.
                # U >= r >= 1.
                # So y, z are positive.
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
