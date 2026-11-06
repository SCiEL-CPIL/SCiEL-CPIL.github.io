import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_max = 10**9
    best_sol = []
    
    # 3-term search
    for a in range(7, 19):
        if time.time() > deadline:
            break
        if 5 * a - 31 <= 0:
            continue
        b_min = a + 1
        b_max = (62 * a) // (5 * a - 31)
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            num = 31 * a * b
            den = 5 * a * b - 31 * b - 31 * a
            if den > 0 and num % den == 0:
                c = num // den
                if c > b and c < best_max:
                    best_max = c
                    best_sol = [a, b, c]
                    
    # If we have time, try 4-term search with max denom < best_max
    if best_sol and time.time() < deadline - 10.0:
        # 4-term: 1/a + 1/b + 1/c + 1/d = 5/31
        # a < b < c < d < best_max
        # a >= 7
        # 4/a > 5/31 => a < 124/5 = 24.8 => a <= 24
        for a in range(7, 25):
            if time.time() > deadline - 5.0:
                break
            # b > a
            # 3/b < 5/31 - 1/a => b > 31a / (5a - 31)
            # Also 1/b < 5/31 - 1/a => b > 31a / (5a - 31) is not the only bound
            # Let's use: 1/a + 1/b + 1/c + 1/d = 5/31
            # 1/b + 1/c + 1/d = 5/31 - 1/a = R_a
            # 3/b > R_a => b < 3 / R_a = 93a / (5a - 31)
            # 1/b < R_a => b > 1 / R_a = 31a / (5a - 31)
            if 5 * a - 31 <= 0:
                continue
            b_min = a + 1
            b_max = (93 * a) // (5 * a - 31)
            
            for b in range(b_min, min(b_max + 1, best_max)):
                if time.time() > deadline - 5.0:
                    break
                R_b = (5 * a - 31) / (31 * a) - 1 / b  # This is float, avoid
                # Use exact fraction logic for c
                # 1/c + 1/d = 5/31 - 1/a - 1/b = R_ab
                # R_ab = (5ab - 31b - 31a) / 31ab
                # Let N = 5ab - 31b - 31a, D = 31ab
                # 1/c + 1/d = N/D
                # (c+d)/cd = N/D => Ncd - Dc - Dd = 0 => (Nc-D)(Nd-D) = D^2
                # This is a standard Diophantine equation.
                # But we can just iterate c.
                # c > b
                # 2/c > N/D => c < 2D/N
                # 1/c < N/D => c > D/N
                if N <= 0:
                    continue
                c_min = b + 1
                c_max = (2 * D) // N
                if c_max >= best_max:
                    c_max = best_max - 1
                    
                for c in range(c_min, c_max + 1):
                    if time.time() > deadline - 5.0:
                        break
                    # 1/d = N/D - 1/c = (Nc - D) / Dc
                    # d = Dc / (Nc - D)
                    num_d = D * c
                    den_d = N * c - D
                    if den_d > 0 and num_d % den_d == 0:
                        d = num_d // den_d
                        if d > c and d < best_max:
                            best_max = d
                            best_sol = [a, b, c, d]
                            # We can break inner loops? No, we want min max.
                            # If we find a smaller d, we update.
                            
    if best_sol:
        return best_sol
    else:
        from fractions import Fraction
        r = Fraction(5, 31)
        values = []
        while r and len(values) < 256:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
