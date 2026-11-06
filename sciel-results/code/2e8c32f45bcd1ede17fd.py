def construct():
    import time
    from fractions import Fraction
    import math
    
    target = Fraction(5, 31)
    deadline = time.time() + 80  # Safety margin for 90s limit
    
    best_max = float('inf')
    best_solution = None
    
    def try_update(solution):
        nonlocal best_max, best_solution
        m = max(solution)
        if m < best_max:
            best_max = m
            best_solution = list(solution)
    
    # 3-term search: 1/a + 1/b + 1/c = 5/31, a < b < c
    # a in [7, 18]
    for a in range(7, 19):
        if time.time() > deadline:
            break
        r = target - Fraction(1, a)
        if r <= 0:
            continue
        n = r.numerator
        d = r.denominator
        # 1/b + 1/c = n/d means (nb-d)(nc-d) = d^2
        # Alternative: iterate b in range and compute c directly
        # b > a, and 1/b < n/d => b > d/n
        # Also 2/b > n/d => b < 2d/n
        b_min = a + 1
        b_max = min(2 * d // n, 10000000)
        if b_max < b_min:
            continue
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            # 1/c = n/d - 1/b = (nb - d) / (db)
            # c = db / (nb - d)
            num = n * b - d
            if num <= 0:
                continue
            den = d * b
            if den % num != 0:
                continue
            c = den // num
            if c <= b:
                continue
            if c > 10000000:
                continue
            if c >= best_max:
                continue
            # Verify
            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                try_update([a, b, c])
    
    # 4-term search: 1/a + 1/b + 1/c + 1/d = 5/31, a < b < c < d
    # a in [7, 24]
    for a in range(7, 25):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            continue
        n1 = r1.numerator
        d1 = r1.denominator
        # 1/b + 1/c + 1/d = r1
        # b > a, and 3/b > r1 => b < 3/r1 = 3*d1/n1
        b_min = a + 1
        b_max = min(3 * d1 // n1, 10000000)
        if b_max < b_min:
            continue
        # Limit b range for performance
        b_upper = min(b_max, a + 500)
        
        for b in range(b_min, b_upper + 1):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, b)
            if r2 <= 0:
                continue
            n2 = r2.numerator
            d2 = r2.denominator
            # 1/c + 1/d = n2/d2
            # c > b, 2/c > n2/d2 => c < 2*d2/n2
            c_min = b + 1
            c_max = min(2 * d2 // n2, 10000000)
            if c_max < c_min:
                continue
            # Limit c range
            c_upper = min(c_max, b + 2000)
            
            for c in range(c_min, c_upper + 1):
                if time.time() > deadline:
                    break
                # 1/d = n2/d2 - 1/c = (n2*c - d2) / (d2*c)
                # d = d2*c / (n2*c - d2)
                num = n2 * c - d2
                if num <= 0:
                    continue
                den = d2 * c
                if den % num != 0:
                    continue
                d = den // num
                if d <= c:
                    continue
                if d > 10000000:
                    continue
                if d >= best_max:
                    continue
                # Verify
                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == target:
                    try_update([a, b, c, d])
    
    # 5-term search: 1/a + 1/b + 1/c + 1/d + 1/e = 5/31, a < b < c < d < e
    # a in [7, 31] (since 5/a > 5/31 => a < 31)
    for a in range(7, 32):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            continue
        n1 = r1.numerator
        d1 = r1.denominator
        # 1/b + 1/c + 1/d + 1/e = r1
        # b > a, 4/b > r1 => b < 4*d1/n1
        b_min = a + 1
        b_max = min(4 * d1 // n1, 10000000)
        if b_max < b_min:
            continue
        b_upper = min(b_max, a + 200)
        
        for b in range(b_min, b_upper + 1):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, b)
            if r2 <= 0:
                continue
            n2 = r2.numerator
            d2 = r2.denominator
            # 1/c + 1/d + 1/e = r2
            # c > b, 3/c > r2 => c < 3*d2/n2
            c_min = b + 1
            c_max = min(3 * d2 // n2, 10000000)
            if c_max < c_min:
                continue
            c_upper = min(c_max, b + 300)
            
            for c in range(c_min, c_upper + 1):
                if time.time() > deadline:
                    break
                r3 = r2 - Fraction(1, c)
                if r3 <= 0:
                    continue
                n3 = r3.numerator
                d3 = r3.denominator
                # 1/d + 1/e = n3/d3
                # d > c, 2/d > n3/d3 => d < 2*d3/n3
                d_min = c + 1
                d_max = min(2 * d3 // n3, 10000000)
                if d_max < d_min:
                    continue
                d_upper = min(d_max, c + 500)
                
                for d in range(d_min, d_upper + 1):
                    if time.time() > deadline:
                        break
                    # 1/e = n3/d3 - 1/d = (n3*d - d3) / (d3*d)
                    # e = d3*d / (n3*d - d3)
                    num = n3 * d - d3
                    if num <= 0:
                        continue
                    den = d3 * d
                    if den % num != 0:
                        continue
                    e = den // num
                    if e <= d:
                        continue
                    if e > 10000000:
                        continue
                    if e >= best_max:
                        continue
                    # Verify
                    if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) + Fraction(1, e) == target:
                        try_update([a, b, c, d, e])
    
    # Fallback
    if best_solution is None:
        r = target
        values = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        best_solution = values
    
    return best_solution
