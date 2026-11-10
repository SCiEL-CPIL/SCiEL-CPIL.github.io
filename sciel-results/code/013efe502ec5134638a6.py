"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    MAX_DENOM = 10**12
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill
    
    answers = [None] * 254  # for n=3..256
    
    def find_solution(n, max_x_limit=500000):
        """Find distinct x,y,z with 1/x+1/y+1/z=4/n using direct search."""
        # WLOG x <= y <= z
        # 4/n = 1/x + 1/y + 1/z
        # 1/x < 4/n => x > n/4
        # 3/x >= 4/n => x <= 3n/4 (since 1/y + 1/z < 2/x when y,z > x)
        # Actually: 1/y + 1/z < 2/x, so 4/n - 1/x < 2/x => 4/n < 3/x => x < 3n/4
        # And 1/x > 4/n - 2/x (since 1/y+1/z < 2/x for y,z > x)
        # More carefully: 4/n = 1/x + 1/y + 1/z > 1/x + 1/x + 1/x = 3/x (if y,z >= x+1)
        # So x > 3n/4... wait let me redo
        # If x <= y <= z, then 1/x >= 1/y >= 1/z
        # 4/n = 1/x + 1/y + 1/z <= 3/x, so x <= 3n/4
        # 4/n = 1/x + 1/y + 1/z >= 1/x + 1/(x+1) + 1/(x+2) approximately
        # Simpler: 1/x < 4/n so x > n/4
        # And 3/x > 4/n (since 1/y + 1/z < 2/x for y,z > x) so x < 3n/4
        
        x_min = n // 4 + 1
        x_max = min((3 * n) // 4, max_x_limit)
        
        for x in range(x_min, x_max + 1):
            # 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
            # Let a = 4x - n, b = nx
            # Need 1/y + 1/z = a/b where a > 0
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # 1/y + 1/z = a/b => b(y+z) = a*y*z => a*y*z - b*y - b*z = 0
            # (a*y - b)(a*z - b) = b^2
            # So we need to find divisors d of b^2 such that d < b^2/d (for y < z)
            # and y = (d + b) / a, z = (b^2/d + b) / a are integers > x
            
            # To avoid factoring b^2 for every x, use a different approach:
            # Iterate over possible y values
            # 1/y < a/b => y > b/a = nx/(4x-n)
            # 1/y > a/b - 1/(x+1) approximately, but simpler bound:
            # Since y <= z, 1/y >= 1/z, so 2/y >= 1/y + 1/z = a/b => y <= 2b/a
            # Also 1/y < a/b => y > b/a
            # So b/a < y <= 2b/a
            
            y_min = b // a + 1
            y_max = 2 * b // a
            
            if y_max < y_min or y_max > 10**8:
                continue
            
            # For each y, check if z is an integer
            # 1/z = a/b - 1/y = (a*y - b) / (b*y)
            # z = b*y / (a*y - b)
            # Need a*y - b > 0 (i.e., y > b/a, which is satisfied by y_min)
            # and (a*y - b) divides b*y
            
            for y in range(y_min, min(y_max + 1, y_min + 100000)):
                num = a * y - b
                if num <= 0:
                    continue
                den = b * y
                if den % num != 0:
                    continue
                z = den // num
                if z <= y:
                    continue
                if z > MAX_DENOM:
                    continue
                # Verify distinctness and equation
                if x == y or x == z or y == z:
                    continue
                # Cross-multiply verification
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    return [x, y, z]
        
        return None
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() - start_time > DEADLINE:
            break
        
        found = find_solution(n)
        
        if found is None:
            # Try with larger x limit
            if time.time() - start_time < DEADLINE - 5:
                found = find_solution(n, max_x_limit=2000000)
        
        answers[idx] = found
    
    return answers
# EVOLVE-BLOCK-END
