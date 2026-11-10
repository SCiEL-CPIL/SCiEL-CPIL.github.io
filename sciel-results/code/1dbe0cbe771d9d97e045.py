import time
import math

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 90s. We aim for 85s.
    deadline = start_time + 85.0
    LIMIT = 10**12
    answers = [None] * 254
    
    def verify(n, x, y, z):
        if x is None or y is None or z is None:
            return False
        if x <= 0 or y <= 0 or z <= 0:
            return False
        if x > LIMIT or y > LIMIT or z > LIMIT:
            return False
        if x == y or y == z or x == z:
            return False
        # 1/x + 1/y + 1/z = 4/n
        # (yz + xz + xy) * n = 4 * x * y * z
        lhs = (y*z + x*z + x*y) * n
        rhs = 4 * x * y * z
        return lhs == rhs

    for n in range(3, 257):
        if time.time() > deadline:
            break
            
        found = None
        
        # x range: n/4 < x <= 3n/4
        # x must be integer.
        # 1/x < 4/n => x > n/4.
        # 3/x >= 4/n => x <= 3n/4.
        x_min = n // 4 + 1
        x_max = 3 * n // 4
        
        # Iterate x. 
        # We can try a few heuristic orders if needed, but sequential is fine given the speed.
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
                
            B = n * x
            
            # We need 1/y + 1/z = A/B
            # (Ay - B)(Az - B) = B^2
            # Let U = Ay - B, V = Az - B.
            # y = (U+B)/A, z = (V+B)/A.
            # Conditions:
            # 1. y > x (distinct, and we assume x <= y <= z for search efficiency, 
            #    though task only requires distinctness. Assuming sorted helps avoid duplicates 
            #    and restricts range).
            # 2. z > y => 1/z < 1/y => Az - B < Ay - B => z < y? No.
            #    1/y + 1/z = A/B.
            #    If y=z, 2/y = A/B => y = 2B/A.
            #    If y < z, then 1/y > 1/z, so 2/y > A/B => y < 2B/A.
            #    Also y > B/A (for z > 0).
            #    So y is in (B/A, 2B/A).
            #    And y > x.
            
            # y_min = max(x + 1, floor(B/A) + 1)
            # y_max = ceil(2B/A) - 1
            
            # Calculate bounds carefully to avoid float issues
            # y > B/A => y >= B//A + 1
            y_lower_bound = B // A + 1
            if B % A == 0:
                # If B/A is integer, y must be strictly greater
                y_lower_bound = B // A + 1
            else:
                y_lower_bound = B // A + 1
            
            y_min = max(x + 1, y_lower_bound)
            
            # y < 2B/A
            # If 2B is divisible by A, y_max = 2B/A - 1
            # Else y_max = floor(2B/A)
            if (2 * B) % A == 0:
                y_max = (2 * B) // A - 1
            else:
                y_max = (2 * B) // A
            
            if y_min > y_max:
                continue
                
            # Iterate y in [y_min, y_max]
            # For each y, check if z is integer and valid.
            # z = B*y / (A*y - B)
            # Denominator D = A*y - B
            # Check if D > 0 (guaranteed by y > B/A)
            # Check if (B*y) % D == 0
            
            for y in range(y_min, y_max + 1):
                if time.time() > deadline:
                    break
                    
                D = A * y - B
                # D should be > 0
                if D <= 0:
                    continue
                    
                if (B * y) % D != 0:
                    continue
                    
                z = (B * y) // D
                
                # Check distinctness and limits
                # We enforced y > x.
                # We enforced y < 2B/A which implies z > y (if y != 2B/A).
                # If y = 2B/A, then z = y, which is excluded by y_max logic above 
                # (if 2B/A is int, y_max is 2B/A - 1).
                # So z > y is guaranteed.
                
                if z > y and z <= LIMIT:
                    # Also need x != z (implied by x < y < z)
                    # and y != z (implied)
                    # and x != y (implied)
                    
                    # Final verification
                    if verify(n, x, y, z):
                        found = [x, y, z]
                        break
            
            if found:
                break
                
        answers[n - 3] = found

    return answers
