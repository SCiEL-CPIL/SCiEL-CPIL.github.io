def construct():
    import time
    from fractions import Fraction

    start_time = time.time()
    deadline = start_time + 85.0

    target = Fraction(5, 31)
    
    # We know from mathematical analysis that 5/31 can be decomposed into 3 terms
    # with a maximum denominator of 248: 1/8 + 1/24 + 1/248? No.
    # Let's verify: 1/8 + 1/24 = 3/24 + 1/24 = 4/24 = 1/6.
    # 5/31 - 1/6 = (30 - 31) / 186 = -1/186. Negative.
    
    # Let's find the actual optimal solution.
    # The known optimal 3-term solution for 5/31 is {8, 24, 248}? No.
    # Let's run a quick search to find the true minimum.
    
    best_solution = None
    best_max = float('inf')

    # 3-term search: 1/a + 1/b + 1/c = 5/31
    # a < b < c
    # a >= ceil(31/5) = 7
    # 3/a >= 5/31 => a <= 93/5 = 18.6 => a <= 18
    
    for a in range(7, 19):
        if time.time() > deadline - 5.0:
            break
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        
        # 1/b + 1/c = rem
        # b > a
        # 2/b >= rem => b <= 2/rem
        # Also 1/b < rem => b > 1/rem
        # b is integer.
        
        # We can solve for b and c exactly.
        # 1/b + 1/c = N/D where rem = N/D
        # (b+c)/(bc) = N/D => D(b+c) = Nbc => Nbc - Db - Dc = 0
        # (Nb - D)(Nc - D) = D^2
        # Let X = Nb - D, Y = Nc - D. XY = D^2.
        # b = (X+D)/N, c = (Y+D)/N.
        # X must be a divisor of D^2.
        # b > a => (X+D)/N > a => X > Na - D.
        # c > b => Y > X.
        
        num, den = rem.numerator, rem.denominator
        # N = num, D = den
        # We need divisors X of D^2 such that:
        # 1. (X + D) % N == 0
        # 2. b = (X + D) // N > a
        # 3. Y = D^2 // X > X
        # 4. (Y + D) % N == 0
        # 5. c = (Y + D) // N > b
        
        D_sq = den * den
        # Iterate divisors of D^2
        # D is small. Max den for a=7 is 31*7=217. D^2 ~ 47k.
        # For a=18, den = 31*18=558. D^2 ~ 311k.
        # Iterating up to sqrt(D^2) = D is fast.
        
        found_for_a = False
        # We want to minimize c. Since c = (Y+D)/N and Y = D^2/X,
        # c is minimized when Y is minimized, which means X is maximized.
        # But X < Y implies X < D.
        # So we want the largest divisor X of D^2 such that X < D and conditions hold.
        # Wait, if we maximize X (closest to D from below), Y is minimized (closest to D from above).
        # This minimizes c.
        
        # Let's just iterate all valid X and keep the one with smallest c.
        min_c_for_a = float('inf')
        best_b_for_a = None
        
        for x in range(1, den + 1):
            if D_sq % x != 0:
                continue
            y = D_sq // x
            if y <= x:
                continue
            if (x + den) % num != 0:
                continue
            b = (x + den) // num
            if b <= a:
                continue
            if (y + den) % num != 0:
                continue
            c = (y + den) // num
            if c <= b:
                continue
            if c < min_c_for_a:
                min_c_for_a = c
                best_b_for_a = b
        
        if best_b_for_a is not None:
            sol = [a, best_b_for_a, min_c_for_a]
            # Verify
            if sum(Fraction(1, d) for d in sol) == target:
                if min_c_for_a < best_max:
                    best_max = min_c_for_a
                    best_solution = sol

    # If we found a 3-term solution, it's likely optimal.
    # The known optimal 3-term solution for 5/31 is indeed small.
    # Let's check if 4-term can beat it.
    # If best_max is found, we check 4-terms with max < best_max.
    
    if best_solution is not None:
        # Check 4-term solutions with max < best_max
        limit = best_max - 1
        # 1/a + 1/b + 1/c + 1/d = 5/31
        # a >= 7
        # a <= 93/5 = 18
        
        for a in range(7, 19):
            if time.time() > deadline - 2.0:
                break
            rem1 = target - Fraction(1, a)
            if rem1 <= 0:
                continue
            
            # 1/b + 1/c + 1/d = rem1
            # b > a
            # 3/b >= rem1 => b <= 3/rem1
            max_b = int(3 / rem1)
            
            for b in range(a + 1, max_b + 1):
                if time.time() > deadline - 1.0:
                    break
                rem2 = rem1 - Fraction(1, b)
                if rem2 <= 0:
                    continue
                
                # 1/c + 1/d = rem2
                # c > b
                # 2/c >= rem2 => c <= 2/rem2
                # Also c < limit
                max_c = min(int(2 / rem2), limit)
                
                num2, den2 = rem2.numerator, rem2.denominator
                D_sq2 = den2 * den2
                
                # Find c, d such that 1/c + 1/d = num2/den2
                # (Nc - D)(Nd - D) = D^2
                # X = Nc - D, Y = Nd - D. XY = D^2.
                # c = (X+D)/N, d = (Y+D)/N.
                # c > b => X > Nb - D.
                # d < limit => (Y+D)/N < limit => Y < N*limit - D.
                
                found_for_b = False
                min_d_for_b = float('inf')
                
                # Iterate divisors X of D_sq2
                # X < Y => X < D2
                for x in range(1, den2 + 1):
                    if D_sq2 % x != 0:
                        continue
                    y = D_sq2 // x
                    if y <= x:
                        continue
                    if (x + den2) % num2 != 0:
                        continue
                    c = (x + den2) // num2
                    if c <= b:
                        continue
                    if (y + den2) % num2 != 0:
                        continue
                    d = (y + den2) // num2
                    if d <= c:
                        continue
                    if d >= limit:
                        continue
                    if d < min_d_for_b:
                        min_d_for_b = d
                
                if min_d_for_b < float('inf'):
                    sol = [a, b, (min_d_for_b + den2) // num2, min_d_for_b] # Wait, c needs to be stored
                    # Recalculate c for the best d? No, we just need the solution.
                    # We found a valid (c, d) pair implicitly by the loop, but we didn't store c.
                    # Let's re-run the loop to store the full solution if it improves best_max.
                    pass # Optimization: we can just check if min_d_for_b < best_max
                    
                    if min_d_for_b < best_max:
                        # Find the corresponding c
                        for x in range(1, den2 + 1):
                            if D_sq2 % x != 0:
                                continue
                            y = D_sq2 // x
                            if y <= x:
                                continue
                            if (x + den2) % num2 != 0:
                                continue
                            c = (x + den2) // num2
                            if c <= b:
                                continue
                            if (y + den2) % num2 != 0:
                                continue
                            d = (y + den2) // num2
                            if d == min_d_for_b:
                                sol = [a, b, c, d]
                                if sum(Fraction(1, dd) for dd in sol) == target:
                                    best_max = min_d_for_b
                                    best_solution = sol
                                break

    # Fallback if no 3 or 4 term found (should not happen)
    if best_solution is None:
        # Greedy expansion fallback
        best_solution = [8, 24, 248] # This is a guess, better to compute
        # Actually, let's just return a known valid one if search fails
        # 5/31 = 1/7 + 1/14 + 1/14 + ... no distinct.
        # 5/31 = 1/8 + 1/28 + 1/1736 is valid.
        best_solution = [8, 28, 1736]
        best_max = 1736

    return best_solution
