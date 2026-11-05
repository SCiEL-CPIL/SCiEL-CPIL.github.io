def construct():
    import time
    from fractions import Fraction
    from math import isqrt

    start_time = time.time()
    # The evaluator kills at 90s. We aim to finish well before to be safe.
    # 85s is a good hard limit, 80s for search.
    deadline = start_time + 85.0

    target = Fraction(5, 31)
    best_solution = None
    best_max = float('inf')

    # Helper to get prime factorization of a number
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
        return divs

    # Pre-compute factorizations for potential denominators in early steps
    # For 1/a + 1/b + ... = 5/31
    # a is small. rem = 5/31 - 1/a = (5a-31)/(31a).
    # Denominator of rem is 31a / gcd(5a-31, 31a).
    # Since 31 is prime and a < 31 (mostly), gcd is usually 1 or small.
    # We will compute divisors on the fly but efficiently.

    def find_2_term_solutions(rem, min_den, max_den_limit):
        """
        Find pairs (c, d) with min_den < c < d <= max_den_limit
        such that 1/c + 1/d = rem.
        Returns list of (c, d) or None if none found.
        Uses divisor generation of D^2.
        """
        num, den = rem.numerator, rem.denominator
        # (num*c - den)(num*d - den) = den^2
        # Let X = num*c - den, Y = num*d - den.
        # XY = den^2.
        # c = (X + den) / num, d = (Y + den) / num.
        # Conditions:
        # 1. X, Y are divisors of den^2.
        # 2. X < Y (since c < d).
        # 3. c > min_den => X > num*min_den - den.
        # 4. d <= max_den_limit => Y <= num*max_den_limit - den.
        # 5. (X + den) % num == 0 and (Y + den) % num == 0.
        
        D_sq = den * den
        # Factorize D_sq. Since den is small (product of small ints), this is fast.
        # den <= 31 * 18 * 18 ~ 10000. D_sq ~ 10^8.
        # Factorizing 10^8 is trivial.
        fac_den = factorize(den)
        # fac_D_sq has exponents doubled
        fac_D_sq = {p: 2*e for p, e in fac_den.items()}
        divs = get_divisors(fac_D_sq)
        
        valid_pairs = []
        
        # Sort divisors to potentially break early? 
        # We want X < Y = D_sq/X. So X < sqrt(D_sq) = den.
        # Also X > num*min_den - den.
        
        # Filter divisors
        lower_x = num * min_den - den
        upper_y = num * max_den_limit - den
        
        for x in divs:
            if x >= den: # Since x < y, x must be < sqrt(D^2) = den
                continue
            if x <= lower_x:
                continue
            
            y = D_sq // x
            
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
            
            valid_pairs.append((c, d))
            
        return valid_pairs

    # Search 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # a >= 7, a <= 18
    # For each a, rem1 = 5/31 - 1/a.
    # Find 1/b + 1/c = rem1 with b > a, c > b.
    # We want to minimize c.
    
    for a in range(7, 19):
        if time.time() > deadline - 5.0:
            break
        
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        
        # b > a
        # 2/b >= rem1 => b <= 2/rem1
        max_b = int(2 / rem1)
        
        for b in range(a + 1, max_b + 1):
            if time.time() > deadline - 2.0:
                break
            
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            
            # c > b
            # 1/c <= rem2 => c >= 1/rem2
            # 1/c > rem2/2 => c < 2/rem2 (not strictly needed for lower bound search, but good for upper)
            # We want to find c, d such that 1/c + 1/d = rem2? No, we are in 3-term mode.
            # Wait, the loop structure above is for 3 terms: 1/a + 1/b + 1/c.
            # So we need to solve 1/b + 1/c = rem1? No.
            # 1/a + 1/b + 1/c = 5/31.
            # rem1 = 5/31 - 1/a.
            # We need 1/b + 1/c = rem1.
            # In the loop `for b in ...`, we pick b.
            # Then we need 1/c = rem1 - 1/b.
            # This is a 1-term solution for c.
            # c = 1 / (rem1 - 1/b).
            # c must be integer and c > b.
            
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            
            if rem2.denominator == 1:
                c = rem2.numerator
                if c > b and c < best_max:
                    # Verify
                    if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                        if c < best_max:
                            best_max = c
                            best_solution = [a, b, c]
            else:
                # c must be integer, so rem2 must be 1/k.
                # rem2 = N/D. We need N/D = 1/k => D = N*k.
                # So D must be divisible by N.
                if rem2.denominator % rem2.numerator == 0:
                    c = rem2.denominator // rem2.numerator
                    if c > b and c < best_max:
                        if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                            if c < best_max:
                                best_max = c
                                best_solution = [a, b, c]

    # If we found a 3-term solution, best_max is likely small (e.g., 248 or similar).
    # Now search for 4-term solutions with max denominator < best_max.
    # 1/a + 1/b + 1/c + 1/d = 5/31.
    # a < b < c < d < best_max.
    
    if best_solution is not None:
        limit = best_max - 1
        
        # a range
        # 4/a >= 5/31 => a <= 124/5 = 24.8 => a <= 24
        # But 1/a < 5/31 => a > 31/5 = 6.2 => a >= 7
        for a in range(7, 25):
            if time.time() > deadline - 5.0:
                break
            
            rem1 = target - Fraction(1, a)
            if rem1 <= 0:
                continue
            
            # 1/b + 1/c + 1/d = rem1
            # b > a
            # 3/b >= rem1 => b <= 3/rem1
            max_b = int(3 / rem1)
            
            for b in range(a + 1, max_b + 1):
                if time.time() > deadline - 3.0:
                    break
                
                rem2 = rem1 - Fraction(1, b)
                if rem2 <= 0:
                    continue
                
                # 1/c + 1/d = rem2
                # c > b
                # d < limit
                # We use find_2_term_solutions(rem2, b, limit)
                
                pairs = find_2_term_solutions(rem2, b, limit)
                if pairs:
                    # We want the pair with smallest d? 
                    # The function returns all valid pairs.
                    # We want to minimize the max denominator, which is d.
                    # So we pick the pair with smallest d.
                    min_d = float('inf')
                    best_c = None
                    for c, d in pairs:
                        if d < min_d:
                            min_d = d
                            best_c = c
                    
                    if min_d < best_max:
                        sol = [a, b, best_c, min_d]
                        # Verify
                        if sum(Fraction(1, x) for x in sol) == target:
                            best_max = min_d
                            best_solution = sol
                            # If we found a 4-term solution, can we find a 5-term with even smaller max?
                            # Probably not worth the complexity given time limits, but let's keep searching for 4-terms
                            # to ensure we have the best 4-term.
        
        # Search 5-term solutions if we have time and best_max is still relatively large?
        # Actually, if best_max is small (e.g., < 100), 5-term search might be expensive.
        # Let's stick to 4-terms unless best_max is large.
        # If best_max is > 100, we might try 5-terms.
        if best_max > 100 and time.time() < deadline - 10.0:
            limit_5 = best_max - 1
            # 1/a + 1/b + 1/c + 1/d + 1/e = 5/31
            # a < b < c < d < e < limit_5
            # a <= 24 (from 5/a >= 5/31 => a <= 31, but tighter bounds exist)
            # 5/a >= 5/31 => a <= 31.
            # 1/a < 5/31 => a > 6.2 => a >= 7.
            
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
                        # e < limit_5
                        pairs = find_2_term_solutions(rem3, c, limit_5)
                        if pairs:
                            min_e = float('inf')
                            best_d = None
                            for d, e in pairs:
                                if e < min_e:
                                    min_e = e
                                    best_d = d
                            
                            if min_e < best_max:
                                sol = [a, b, c, best_d, min_e]
                                if sum(Fraction(1, x) for x in sol) == target:
                                    best_max = min_e
                                    best_solution = sol

    # Fallback if nothing found (should not happen)
    if best_solution is None:
        # Known valid solution
        best_solution = [8, 28, 1736]
        best_max = 1736

    return best_solution
