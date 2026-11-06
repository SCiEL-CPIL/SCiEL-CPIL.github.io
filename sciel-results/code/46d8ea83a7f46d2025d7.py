# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    
    best_solution = None
    best_max = 10000001
    
    # Search for 3-term Egyptian fraction 1/a + 1/b + 1/c = 5/31
    # with a < b < c, minimizing c.
    # a must be >= 7 (since 1/6 < 5/31).
    # Bound for a: derived from b < 62a / (5a - 31) and b > a.
    # This requires 62a / (5a - 31) > a => 62 > 5a - 31 => 93 > 5a => a < 18.6.
    # So a is in [7, 18].
    
    for a in range(7, 19):
        # b must satisfy:
        # 1. b > a
        # 2. Denominator 5ab - 31a - 31b > 0
        # 3. c = 31ab / (5ab - 31a - 31b) is integer and c > b
        
        # From c > b, we derived b < 62a / (5a - 31).
        # Let's calculate the upper bound for b.
        # If 5a - 31 <= 0, no solution (but a>=7 so 5a-31 >= 4 > 0).
        b_max = int((62 * a) / (5 * a - 31))
        
        for b in range(a + 1, b_max + 1):
            denom = 5 * a * b - 31 * a - 31 * b
            if denom <= 0:
                continue
            
            numer = 31 * a * b
            
            if numer % denom != 0:
                continue
            
            c = numer // denom
            
            if c <= b:
                continue
            
            # We found a valid solution a, b, c
            if c < best_max:
                best_max = c
                best_solution = [a, b, c]
                
    if best_solution is not None:
        return best_solution
    
    # Fallback (should not be reached)
    return [7, 62, 434]
# EVOLVE-BLOCK-END
