"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    
    # Search for Egyptian fraction of 5/31 minimizing the largest denominator
    # Try 3 terms and 4 terms
    
    best_solution = None
    best_max = 10000001
    
    # --- Search 3 terms ---
    for a in range(7, 500):
        p = 5 * a - 31
        q = 31 * a
        if p <= 0:
            continue
        
        for b in range(a + 1, 5000):
            denom_val = p * b - q
            if denom_val <= 0:
                continue
            
            numerator_val = q * b
            
            if numerator_val % denom_val != 0:
                continue
            
            c = numerator_val // denom_val
            
            if c <= b:
                continue
            
            if c < best_max:
                best_max = c
                best_solution = [a, b, c]
                
    # --- Search 4 terms ---
    # We look for solutions with max denominator < best_max
    # Iterate a, b, c and solve for d
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # 1/d = 5/31 - 1/a - 1/b - 1/c
    # d = 1 / (5/31 - 1/a - 1/b - 1/c)
    
    # To keep it fast, limit a, b, c
    # a must be at least 7
    # If d < best_max, then 1/d > 1/best_max
    # So 5/31 - 1/a - 1/b - 1/c > 1/best_max
    # This implies 1/a + 1/b + 1/c < 5/31 - 1/best_max
    
    for a in range(7, 100):
        rem_a = Fraction(5, 31) - Fraction(1, a)
        if rem_a <= 0:
            continue
        for b in range(a + 1, 200):
            rem_ab = rem_a - Fraction(1, b)
            if rem_ab <= 0:
                continue
            for c in range(b + 1, 300):
                rem_abc = rem_ab - Fraction(1, c)
                if rem_abc <= 0:
                    continue
                
                # Check if 1/rem_abc is an integer d
                # d = 1 / rem_abc
                # rem_abc = num / den
                # d = den / num
                # Must be integer and d > c
                
                num = rem_abc.numerator
                den = rem_abc.denominator
                
                if den % num != 0:
                    continue
                
                d = den // num
                
                if d <= c:
                    continue
                
                # Check distinctness (guaranteed by a<b<c<d)
                # Check range
                if d > 10000000:
                    continue
                
                if d < best_max:
                    best_max = d
                    best_solution = [a, b, c, d]
    
    if best_solution is not None:
        return best_solution
    
    # Fallback
    return [7, 62, 434]
# EVOLVE-BLOCK-END
