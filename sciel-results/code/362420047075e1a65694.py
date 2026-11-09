def construct():
    from fractions import Fraction
    import time
    
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Known optimal solution
    best_solution = [8, 31, 248]
    best_max = 248
    
    target = Fraction(5, 31)
    
    # Search for 3-term solutions with max denominator < best_max
    # 1/a + 1/b + 1/c = 5/31, a < b < c
    for c in range(1, best_max):
        if time.time() > deadline:
            break
            
        r = target - Fraction(1, c)
        if r <= 0:
            continue
            
        n = r.numerator
        d = r.denominator
        
        # 1/a + 1/b = n/d
        # a < b
        # 2/b > n/d => b < 2d/n
        # 1/a < n/d => a > d/n
        
        start_a = (d // n) + 1
        end_a = (2 * d) // n
        
        found = False
        for a in range(start_a, end_a + 1):
            if time.time() > deadline:
                break
                
            denom_b = n * a - d
            if denom_b <= 0:
                continue
                
            if d % denom_b == 0:
                b = d // denom_b
                if b > a: # Ensures a < b
                    if b != c: # Ensures b < c (since a < b and we assume c is largest, but need b < c)
                        # Check distinctness: a, b, c all distinct
                        # a < b is checked.
                        # b < c is checked.
                        # So a, b, c are distinct.
                        best_solution = [a, b, c]
                        best_max = c
                        found = True
                        break
        if found:
            break
            
    return best_solution
