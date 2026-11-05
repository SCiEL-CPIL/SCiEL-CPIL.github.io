def construct():
    from fractions import Fraction
    import time
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    target = Fraction(5, 31)
    best_max = float('inf')
    best_solution = None
    
    def update_best(denoms):
        nonlocal best_max, best_solution
        max_den = max(denoms)
        if max_den < best_max:
            best_max = max_den
            best_solution = list(denoms)
    
    # Try 3-term solutions
    for a in range(7, 200):
        if time.time() > deadline:
            break
        r_ab = target - Fraction(1, a)
        if r_ab <= 0:
            continue
        b_min = max(a + 1, (r_ab.denominator + r_ab.numerator - 1) // r_ab.numerator)
        for b in range(b_min, b_min + 5000):
            if time.time() > deadline:
                break
            r_c = r_ab - Fraction(1, b)
            if r_c <= 0:
                break
            if r_c.denominator % r_c.numerator == 0:
                c = r_c.denominator // r_c.numerator
                if c > b and c < 10000000:
                    update_best([a, b, c])
    
    # Try 4-term solutions
    for a in range(7, 30):
        if time.time() > deadline:
            break
        r_ab = target - Fraction(1, a)
        if r_ab <= 0:
            continue
        b_min = max(a + 1, (r_ab.denominator + r_ab.numerator - 1) // r_ab.numerator)
        for b in range(b_min, b_min + 3000):
            if time.time() > deadline:
                break
            r_bc = r_ab - Fraction(1, b)
            if r_bc <= 0:
                break
            c_min = max(b + 1, (r_bc.denominator + r_bc.numerator - 1) // r_bc.numerator)
            for c in range(c_min, c_min + 3000):
                if time.time() > deadline:
                    break
                r_d = r_bc - Fraction(1, c)
                if r_d <= 0:
                    break
                if r_d.denominator % r_d.numerator == 0:
                    d = r_d.denominator // r_d.numerator
                    if d > c and d < 10000000:
                        update_best([a, b, c, d])
    
    if best_solution is not None:
        return best_solution
    
    # Fallback
    r = target
    values = []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
