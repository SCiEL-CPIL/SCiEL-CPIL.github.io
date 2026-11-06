import time

def construct():
    """
    Construct distinct three-term unit-fraction representations of 4/n for n=3..256.
    Returns 254 rows; each is None or [x, y, z] with x<y<z, all distinct, <= 10^12.
    """
    deadline = time.time() + 80  # 80s safety margin within 90s wall-clock
    MAX_DEN = 10**12
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # We need 1/x + 1/y + 1/z = 4/n with x < y < z (distinct positive integers)
        # Since x < y < z, we have 3/x > 4/n => x < 3n/4
        # And 1/x < 4/n => x > n/4
        # So x is in range [floor(n/4)+1, floor(3n/4)]
        
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        
        for x in range(start_x, end_x + 1):
            if time.time() > deadline:
                break
            
            # After fixing x: 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
            # Let A = 4x - n, B = nx
            # We need 1/y + 1/z = A/B => (y+z)/(yz) = A/B => B(y+z) = A*y*z
            # => A*y*z - B*y - B*z = 0
            # => (A*y - B)(A*z - B) = B^2
            # Let u = A*y - B, v = A*z - B, then u*v = B^2
            # y = (u+B)/A, z = (v+B)/A
            # Need u,v positive divisors of B^2, u <= v (since y <= z)
            # Also need y > x, z > y, all distinct, all <= MAX_DEN
            # u must divide B^2, and (u+B) % A == 0, (B^2//u + B) % A == 0
            
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
            
            # y must be > x, so 1/y < 1/x, meaning y > x
            # From 1/y + 1/z = A/B, since z > y, we have 2/y > A/B => y < 2B/A
            # And 1/y < A/B => y > B/A
            # Also y > x
            # So y is in range [max(x+1, B//A + 1), floor(2B/A)]
            
            y_min = max(x + 1, B // A + 1)
            y_max = 2 * B // A
            
            # To ensure y < z, we need y <= (B + sqrt(B^2 + A*B*... )) 
            # Actually from (A*y - B)(A*z - B) = B^2, with z > y means A*z - B > A*y - B
            # So u < v, i.e., u < B^2/u, i.e., u^2 < B^2, i.e., u < B
            # So A*y - B < B => A*y < 2B => y < 2B/A, which is our y_max bound
            # And u > 0 means A*y > B means y > B/A, which is our y_min bound
            
            # For each valid y in range, check if z = B*y / (A*y - B) is integer
            # z = B*y / (A*y - B)
            # Need (A*y - B) divides B*y
            # Let d = A*y - B. Need d | B*y. Since d = A*y - B, we can check if B*y % d == 0
            
            for y in range(y_min, y_max + 1):
                d = A * y - B
                if d <= 0:
                    continue
                if B * y % d == 0:
                    z = B * y // d
                    if z > y and z <= MAX_DEN and y <= MAX_DEN and x <= MAX_DEN:
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
