def construct():
    import time
    from fractions import Fraction
    from math import isqrt

    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    target = Fraction(5, 31)
    best_solution = None
    best_max = float('inf')

    # Helper to factorize a number
    def factorize(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    # Helper to generate all divisors of n from its prime factorization
    def get_divisors(factors):
        divs = [1]
        for p, exp in factors.items():
            new_divs = []
            for d in divs:
                val = d
                for _ in range(exp):
                    val *= p
                    new_divs.append(val)
            divs += new_divs
        return sorted(divs)

    # Find 2-term solutions (c, d) such that 1/c + 1/d = rem, with min_den < c < d <= max_den_limit
    # Returns the pair with the smallest d, or None
    def find_2_term_solutions(rem, min_den, max_den_limit):
        num, den = rem.numerator, rem.denominator
        D_sq = den * den
        
        # (num*c - den)(num*d - den) = den^2
        # X = num*c - den, Y = num*d - den
        # XY = D_sq
        # c = (X + den) / num
        # d = (Y + den) / num
        
        fac_den = factorize(den)
        fac_D_sq = {p: 2 * e for p, e in fac_den.items()}
        divs = get_divisors(fac_D_sq)
        
        lower_x = num * min_den - den
        upper_y = num * max_den_limit - den
        
        best_d = float('inf')
        best_pair = None
        
        for x in divs:
            if x >= den:
                continue
            if x <= lower_x:
                continue
            
            y = D_sq // x
            if y <= x:
                continue
            if y > upper_y:
                continue
            
            if (x + den) % num != 0:
                continue
            c = (x + den) // num
            if c <= min_den:
                continue
            
            if (y + den) % num != 0:
                continue
            d = (y + den) // num
            if d <= c:
                continue
            if d > max_den_limit:
                continue
            
            if d < best_d:
                best_d = d
                best_pair = (c, d)
                
        return best_pair

    # Search for solutions with a given number of terms, minimizing max denominator
    # We'll search for 3, 4, 5 term solutions

    # First, establish a baseline with known solutions
    baseline_solutions = [
        [8, 31, 248],      # 3-term, max=248
        [7, 62, 434],      # 3-term, max=434
    ]
    for sol in baseline_solutions:
        if len(set(sol)) == len(sol):
            s = sum(Fraction(1, x) for x in sol)
            if s == target:
                mx = max(sol)
                if mx < best_max:
                    best_max = mx
                    best_solution = sol

    # Search 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # a < b < c
    # 1/a < 5/31 => a > 31/5 = 6.2 => a >= 7
    # 3/a >= 5/31 => a <= 93/5 = 18.6 => a <= 18
    
    if time.time() < deadline - 20.0:
        for a in range(7, 19):
            if time.time() > deadline - 15.0:
                break
                
            rem1 = target - Fraction(1, a)
            if rem1 <= 0:
                continue
            
            # 1/b + 1/c = rem1
            # b > a
            # 2/b >= rem1 => b <= 2/rem1
            # 1/b < rem1 => b > 1/rem1
            max_b = int(2 / rem1)
            
            for b in range(a + 1, max_b + 1):
                if time.time() > deadline - 10.0:
                    break
                    
                rem2 = rem1 - Fraction(1, b)
                if rem2 <= 0:
                    continue
                
                # We need 1/c = rem2. So c = 1/rem2.
                # c must be integer.
                if rem2.denominator % rem2.numerator == 0:
                    c = rem2.denominator // rem2.numerator
                    if c > b:
                        sol = [a, b, c]
                        if sum(Fraction(1, d) for d in sol) == target:
                            if c < best_max:
                                best_max = c
                                best_solution = sol

    # Search 4-term solutions: 1/a + 1/b + 1/c + 1/d = 5/31
    # a < b < c < d
    # 4/a >= 5/31 => a <= 124/5 = 24.8 => a <= 24
    # 1/a < 5/31 => a >= 7
    
    if best_solution is not None and time.time() < deadline - 15.0:
        limit = best_max - 1
        
        for a in range(7, 25):
            if time.time() > deadline - 10.0:
                break
                
            rem1 = target - Fraction(1, a)
            if rem1 <= 0:
                continue
            
            # 1/b + 1/c + 1/d = rem1
            # b > a
            # 3/b >= rem1 => b <= 3/rem1
            max_b = int(3 / rem1)
            
            for b in range(a + 1, max_b + 1):
                if time.time() > deadline - 5.0:
                    break
                    
                rem2 = rem1 - Fraction(1, b)
                if rem2 <= 0:
                    continue
                
                # 1/c + 1/d = rem2
                # c > b
                # d < limit
                pair = find_2_term_solutions(rem2, b, limit)
                if pair:
                    c, d = pair
                    sol = [a, b, c, d]
                    if sum(Fraction(1, dd) for dd in sol) == target:
                        if d < best_max:
                            best_max = d
                            best_solution = sol

    # Search 5-term solutions if we have room in time and best_max is still relatively high
    if best_solution is not None and best_max > 100 and time.time() < deadline - 10.0:
        limit = best_max - 1
        
        # 5/a >= 5/31 => a <= 31
        # 1/a < 5/31 => a >= 7
        
        for a in range(7, 32):
            if time.time() > deadline - 5.0:
                break
                
            rem1 = target - Fraction(1, a)
            if rem1 <= 0:
                continue
            
            # 1/b + 1/c + 1/d + 1/e = rem1
            # b > a
            # 4/b >= rem1 => b <= 4/rem1
            max_b = int(4 / rem1)
            
            for b in range(a + 1, max_b + 1):
                if time.time() > deadline - 2.0:
                    break
                    
                rem2 = rem1 - Fraction(1, b)
                if rem2 <= 0:
                    continue
                
                # 1/c + 1/d + 1/e = rem2
                # c > b
                # 3/c >= rem2 => c <= 3/rem2
                max_c = int(3 / rem2)
                
                for c in range(b + 1, max_c + 1):
                    if time.time() > deadline - 1.0:
                        break
                        
                    rem3 = rem2 - Fraction(1, c)
                    if rem3 <= 0:
                        continue
                    
                    # 1/d + 1/e = rem3
                    # d > c
                    # e < limit
                    pair = find_2_term_solutions(rem3, c, limit)
                    if pair:
                        d, e = pair
                        sol = [a, b, c, d, e]
                        if sum(Fraction(1, dd) for dd in sol) == target:
                            if e < best_max:
                                best_max = e
                                best_solution = sol

    # Final polish: try to find even better solutions with tighter bounds
    # If we found a solution, try to improve it further
    if best_solution is not None and time.time() < deadline - 5.0:
        # Try 4-term solutions with tighter limit
        limit = best_max - 1
        if limit > 20:
            for a in range(7, min(25, limit)):
                if time.time() > deadline - 2.0:
                    break
                    
                rem1 = target - Fraction(1, a)
                if rem1 <= 0:
                    continue
                
                max_b = int(3 / rem1)
                max_b = min(max_b, limit - 2)
                
                for b in range(a + 1, max_b + 1):
                    if time.time() > deadline - 1.0:
                        break
                        
                    rem2 = rem1 - Fraction(1, b)
                    if rem2 <= 0:
                        continue
                    
                    pair = find_2_term_solutions(rem2, b, limit)
                    if pair:
                        c, d = pair
                        sol = [a, b, c, d]
                        if sum(Fraction(1, dd) for dd in sol) == target:
                            if d < best_max:
                                best_max = d
                                best_solution = sol

    # Fallback
    if best_solution is None:
        best_solution = [8, 31, 248]
        best_max = 248

    return best_solution
