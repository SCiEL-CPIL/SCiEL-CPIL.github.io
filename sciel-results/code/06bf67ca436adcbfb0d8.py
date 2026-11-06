import time
import math

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254
    BOUND = 10**12
    
    def find_solution(n, x_start, x_end, y_div_limit=200, check_time=True):
        """Find a distinct unit fraction triple for 1/x+1/y+1/z=4/n."""
        for x in range(x_start, x_end + 1):
            if check_time and time.time() > deadline - 3.0:
                return None
            
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            y_low_1 = b // a + 1
            y_low_3 = x + 1
            
            denom = BOUND * a - b
            if denom <= 0:
                continue
            y_low_4 = (BOUND * b + denom - 1) // denom
            
            y_low = max(y_low_1, y_low_3, y_low_4)
            y_high = (2 * b + a - 1) // a - 1
            
            if y_high < y_low:
                continue
            
            # Try small consecutive y values first
            for y in range(y_low, min(y_low + 10, y_high + 1)):
                z_num = b * y
                z_den = a * y - b
                if z_den <= 0:
                    continue
                if z_num % z_den == 0:
                    z = z_num // z_den
                    if z > y and y > x and z <= BOUND and y <= BOUND and x <= BOUND:
                        # Verify with cross multiplication
                        if n * y * z + n * x * z + n * x * y == 4 * x * y * z:
                            return [x, y, z]
            
            # Try divisor-based y finding: y = (d + b) / a where d | b
            limit = min(y_div_limit, b)
            for d in range(1, limit + 1):
                if b % d == 0:
                    if (d + b) % a == 0:
                        y = (d + b) // a
                        if y_low <= y <= y_high and y > x:
                            z_num = b * y
                            z_den = a * y - b
                            if z_den > 0 and z_num % z_den == 0:
                                z = z_num // z_den
                                if z > y and z <= BOUND:
                                    if n * y * z + n * x * z + n * x * y == 4 * x * y * z:
                                        return [x, y, z]
                    # Also try the complementary divisor
                    d2 = b // d
                    if d2 != d:
                        if (d2 + b) % a == 0:
                            y = (d2 + b) // a
                            if y_low <= y <= y_high and y > x:
                                z_num = b * y
                                z_den = a * y - b
                                if z_den > 0 and z_num % z_den == 0:
                                    z = z_num // z_den
                                    if z > y and z <= BOUND:
                                        if n * y * z + n * x * z + n * x * y == 4 * x * y * z:
                                            return [x, y, z]
            
            # Also try d values where d ≡ -b (mod a) even if d doesn't divide b
            # This can sometimes yield solutions
            for offset in range(1, 50):
                d_candidate = b + offset * a
                if d_candidate > 0 and d_candidate <= limit:
                    if (d_candidate + b) % a == 0:
                        y = (d_candidate + b) // a
                        if y_low <= y <= y_high and y > x:
                            z_num = b * y
                            z_den = a * y - b
                            if z_den > 0 and z_num % z_den == 0:
                                z = z_num // z_den
                                if z > y and z <= BOUND:
                                    if n * y * z + n * x * z + n * x * y == 4 * x * y * z:
                                        return [x, y, z]
        
        return None
    
    # First pass: quick search for each n
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline - 5.0:
            break
        
        x_min = n // 4 + 1
        found = find_solution(n, x_min, x_min + 500, y_div_limit=100)
        answers[idx] = found
    
    # Second pass: extended search for unsolved n's
    for n in range(3, 257):
        idx = n - 3
        if answers[idx] is not None:
            continue
        if time.time() > deadline - 10.0:
            break
        
        x_min = n // 4 + 1
        found = find_solution(n, x_min, x_min + 3000, y_div_limit=500)
        answers[idx] = found
    
    # Third pass: even more extended search for remaining unsolved n's
    for n in range(3, 257):
        idx = n - 3
        if answers[idx] is not None:
            continue
        if time.time() > deadline - 2.0:
            break
        
        x_min = n // 4 + 1
        found = find_solution(n, x_min, x_min + 10000, y_div_limit=1000)
        answers[idx] = found
    
    return answers
