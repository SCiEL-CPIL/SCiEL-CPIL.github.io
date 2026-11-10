import time
import math

def construct():
    # Deadline guard with safety margin
    start_time = time.time()
    deadline = start_time + 85
    
    answers = []
    MAX_Z = 10**12
    n_total = 254  # n from 3 to 256 inclusive
    
    # Precompute some useful values
    # For each n, we'll try to find x, y, z such that 1/x + 1/y + 1/z = 4/n
    # with x < y < z, all distinct, and <= MAX_Z
    
    for idx, n in enumerate(range(3, 257)):
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        remaining_n = n_total - idx
        
        if remaining <= 0:
            answers.append(None)
            continue
        
        # Allocate time for this n
        time_per_n = min(remaining / remaining_n, 0.35)
        n_deadline = time.time() + time_per_n
        
        found = None
        
        # x must satisfy: x > n/4 and x <= 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Try x values where A = 4x - n is large (x near 3n/4) first,
        # since this gives smaller y-ranges
        x_values = list(range(x_max, x_min - 1, -1))
        
        # Also try some small x values (near n/4) as fallback
        # but prioritize larger x
        for x in x_values:
            if time.time() > n_deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
            
            B = n * x
            
            # y > B/A and y < (n+B)/A for z > y
            y_min = B // A + 1
            y_max = (n + B) // A
            
            # Ensure y > x for distinctness (assuming x < y < z)
            if y_min <= x:
                y_min = x + 1
                
            if y_max < y_min:
                continue
            
            # Iterate y, but limit the number of iterations
            # to avoid spending too much time on a single x
            y_count = 0
            max_y_iters = 2000  # Limit iterations per x
            
            for y in range(y_min, y_max + 1):
                y_count += 1
                if y_count > max_y_iters:
                    break
                
                # z = B*y / (A*y - B)
                D = A * y - B
                if D <= 0:
                    continue
                
                if (B * y) % D != 0:
                    continue
                    
                z = (B * y) // D
                
                # Check distinctness and bounds
                if z <= y:
                    continue
                if z > MAX_Z:
                    continue
                
                # x < y < z ensures distinctness
                found = [x, y, z]
                break
            
            if found:
                break
        
        # If not found with the above, try a different strategy:
        # Use the identity (Ay - B)(Az - B) = B^2
        # For each x, enumerate divisors of B^2
        if found is None and time.time() < n_deadline:
            # Try a few x values with the divisor approach
            for x in range(x_max, max(x_min, x_max - 20), -1):
                if time.time() > n_deadline:
                    break
                    
                A = 4 * x - n
                if A <= 0:
                    continue
                
                B = n * x
                B_sq = B * B
                
                # Find divisors d of B^2 such that d > 0 and d <= B
                # and (d + B) % A == 0 and (B_sq // d + B) % A == 0
                # We can't enumerate all divisors of B^2 easily, so skip this for now
                
                # Instead, try y values near B/A
                y_min = B // A + 1
                y_max = (n + B) // A
                if y_min <= x:
                    y_min = x + 1
                if y_max < y_min:
                    continue
                
                # Try a limited range around B/A
                y_center = B // A
                y_start = max(y_min, y_center)
                y_end = min(y_max, y_center + 100)
                
                for y in range(y_start, y_end + 1):
                    D = A * y - B
                    if D <= 0:
                        continue
                    if (B * y) % D != 0:
                        continue
                    z = (B * y) // D
                    if z <= y or z > MAX_Z:
                        continue
                    found = [x, y, z]
                    break
                if found:
                    break
        
        answers.append(found)
        
    return answers
