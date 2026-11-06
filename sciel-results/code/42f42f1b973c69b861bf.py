import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill
    
    answers = [None] * 254
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
            
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Strategy: Iterate x, then iterate u = A*y - B in steps of A
        # u must divide B^2.
        # y = (u+B)/A, z = (B^2/u + B)/A
        # Constraints: y > x, z > y, x,y,z distinct, <= 10^12
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
                
            B = n * x
            
            # u = A*y - B
            # y > x => u > A*x - B
            # y <= z => u <= B (since u*v=B^2 and u<=v => u<=B)
            # Also y >= 1 => u >= A - B. But y>x ensures u > Ax-B.
            # We iterate u such that u = A*y - B.
            # This means u = A*y - B => u + B = A*y => u = -B mod A.
            # Let u = k*A - B.
            # We need u > A*x - B => k*A - B > A*x - B => k > x.
            # So k starts from x+1.
            # We need u <= B => k*A - B <= B => k*A <= 2B => k <= 2B/A.
            
            k_min = x + 1
            k_max = (2 * B) // A
            
            if k_max < k_min:
                continue
            
            # Iterate k
            # u = k*A - B
            # Check if u divides B^2
            # If yes, v = B^2 // u
            # y = (u + B) // A = k
            # z = (v + B) // A
            # Check z > y (k) and z <= 10^12
            
            for k in range(k_min, k_max + 1):
                u = k * A - B
                # u is positive because k > x >= 1 and A > 0, B = nx.
                # Check divisibility
                if (B * B) % u != 0:
                    continue
                
                v = (B * B) // u
                # y = k
                y = k
                # z = (v + B) // A
                # We need (v+B) to be divisible by A
                if (v + B) % A != 0:
                    continue
                
                z = (v + B) // A
                
                # Check distinctness and bounds
                if z <= y:
                    continue
                if z > 1000000000000:
                    continue
                # x, y, z distinct?
                # y = k > x (since k_min = x+1)
                # z > y
                # So x < y < z. Distinctness holds.
                
                # Final verification
                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    found = [x, y, z]
                    break
            
            if found:
                break
                
        answers[idx] = found
        
    return answers
