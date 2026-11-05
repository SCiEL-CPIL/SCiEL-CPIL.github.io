def construct():
    from fractions import Fraction
    import time
    
    deadline = time.time() + 85  # 85s deadline with 5s margin
    
    target = Fraction(5, 31)
    
    # Try to find solutions with small maximum denominator
    # Start with 3-term solutions, then 4-term, etc.
    
    best_solution = None
    best_max = 10000001
    
    # Search for 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # For given a, b: c = 31ab / (5ab - 31a - 31b)
    # Need 5ab - 31a - 31b > 0 and divides 31ab
    
    # Try a from 7 upward (since 1/7 ≈ 0.143 < 5/31 ≈ 0.161, but we need sum to be 5/31)
    # Actually 1/a < 5/31, so a > 31/5 = 6.2, so a >= 7
    
    # For 3 terms, let's search a, b with a < b, and compute c
    # To keep max small, try small a, b first
    
    # Strategy: iterate over possible max values and try to find solutions
    # But more efficient: iterate a, b in increasing order and check if c is valid integer
    
    found_3term = False
    for a in range(7, 200):
        if time.time() > deadline:
            break
        for b in range(a + 1, 500):
            if time.time() > deadline:
                break
            # Compute numerator and denominator for c
            # 1/c = 5/31 - 1/a - 1/b = (5ab - 31b - 31a) / (31ab)
            num = 5 * a * b - 31 * a - 31 * b
            den = 31 * a * b
            if num <= 0:
                continue
            # c = den / num, must be integer
            if den % num == 0:
                c = den // num
                if c > b and c != a and c != b:
                    # Check distinctness and validity
                    if len({a, b, c}) == 3:
                        solution = [a, b, c]
                        max_val = max(solution)
                        if max_val < best_max:
                            best_max = max_val
                            best_solution = solution
                        found_3term = True
                        if max_val < 100:
                            return solution
    
    if found_3term and best_solution:
        return best_solution
    
    # Try 4-term solutions if 3-term not found or want to verify
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # Fix a, b, c, solve for d
    # This is more complex, so let's do a limited search
    
    # For 4 terms, fix a, b, c and compute d
    for a in range(7, 30):
        if time.time() > deadline:
            break
        for b in range(a + 1, 50):
            if time.time() > deadline:
                break
            for c in range(b + 1, 80):
                if time.time() > deadline:
                    break
                # 1/d = 5/31 - 1/a - 1/b - 1/c
                # = (5abc - 31bc - 31ac - 31ab) / (31abc)
                num = 5 * a * b * c - 31 * b * c - 31 * a * c - 31 * a * b
                den = 31 * a * b * c
                if num <= 0:
                    continue
                if den % num == 0:
                    d = den // num
                    if d > c and len({a, b, c, d}) == 4:
                        solution = [a, b, c, d]
                        max_val = max(solution)
                        if max_val < best_max:
                            best_max = max_val
                            best_solution = solution
    
    if best_solution:
        return best_solution
    
    # Fallback: use the greedy approach from the seed
    r, values = Fraction(5, 31) - Fraction(1, 8), [8]
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
