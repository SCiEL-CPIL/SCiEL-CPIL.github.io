"""Improved program: search multiple x values and use divisibility trick for z."""
import math
import time

def construct():
    answers = []
    deadline = time.time() + 80  # safety margin before 90s kill
    BOUND = 10**12
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must satisfy 1/x < 4/n => x > n/4
        # Also 1/x + 1/y + 1/z = 4/n with x<=y<=z means 3/x >= 4/n => x <= 3n/4
        # But we need x < y < z, so let's search a reasonable range of x
        
        start_x = n // 4 + 1
        # Search multiple x values; small x tends to work well
        # Upper bound: we need A = 4x - n > 0, and we need solutions to exist
        # For x too large, A is large and the search space for y becomes constrained
        # Let's try up to start_x + 200 or so, but with time check
        
        max_x_search = min(start_x + 300, 3 * n // 4 + 10)
        
        for x in range(start_x, max_x_search + 1):
            if time.time() > deadline:
                break
            
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # We need: 1/y + 1/z = A/B
            # => z = B*y / (A*y - B)
            # For z to be integer, (A*y - B) must divide B*y
            # Let d = A*y - B, so y = (B + d)/A, and we need d | B*(B+d)/A
            # Equivalently: d | B^2/A (if A | B^2) or more generally:
            # d must be a positive divisor of B*y where y = (B+d)/A
            # So d | B*(B+d)/A, meaning A*d | B*(B+d)
            # Let's just iterate over possible y values
            
            # y must be > x (distinct, and we'll enforce x<y<z)
            # Also 1/y < A/B => y > B/A
            # And 2/y > A/B => y < 2B/A (since z > y means 1/z < 1/y, so 1/y > A/(2B))
            # Wait: 1/y + 1/z = A/B with y < z means 2/y > A/B => y < 2B/A
            # And 1/y < A/B => y > B/A
            
            y_min = max(x + 1, B // A + 1)
            y_max = 2 * B // A + 1
            
            # But y_max - y_min could be large. Let's use the divisor approach instead.
            # z = B*y / (A*y - B). Let denom = A*y - B.
            # We need denom > 0 and denom | B*y.
            # Since denom = A*y - B, we have y = (denom + B) / A.
            # So denom must satisfy: denom ≡ -B (mod A), i.e., denom = A*k - B for some k
            # And denom | B*(A*k - B)/A... this is getting complex.
            
            # Simpler: iterate y in a bounded range, but limit iterations
            # The range y_min to y_max might be large for large x.
            # Let's cap the number of y iterations per x.
            
            y_range_size = y_max - y_min + 1
            if y_range_size < 0:
                continue
            
            # For small y_range, iterate directly. For large, use divisor method.
            if y_range_size <= 5000:
                for y in range(y_min, y_max + 1):
                    denom = A * y - B
                    if denom <= 0:
                        continue
                    if B * y % denom == 0:
                        z = B * y // denom
                        if z > y and z <= BOUND:
                            # Check distinctness
                            if x != y and y != z and x != z:
                                found = [x, y, z]
                                break
            else:
                # Use divisor-based search
                # We need A*y - B | B*y
                # Let d = A*y - B > 0, so y = (B + d)/A, requiring (B+d) % A == 0
                # And d | B*(B+d)/A
                # d must be a divisor of B*(B+d)/A... circular.
                # Alternative: d | B*y and y = (B+d)/A
                # So d | B*(B+d)/A => A*d | B*(B+d)
                # Let g = gcd(A, B). Then A/g * d | (B/g)*(B+d)
                # This is complex. Let's just iterate over divisors of B^2/A if A|B^2,
                # or fall back to limited y iteration.
                
                # Fallback: iterate y but with a cap
                for y in range(y_min, min(y_min + 5000, y_max + 1)):
                    denom = A * y - B
                    if denom <= 0:
                        continue
                    if B * y % denom == 0:
                        z = B * y // denom
                        if z > y and z <= BOUND:
                            if x != y and y != z and x != z:
                                found = [x, y, z]
                                break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
