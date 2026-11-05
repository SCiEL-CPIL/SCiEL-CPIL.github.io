def construct():
    import time
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    target_num = 5
    target_den = 31
    
    # Known good solution: 5/31 = 1/8 + 1/28 + 1/1736
    # Max is 1736. Score = 1.0 implies this is the baseline or current best.
    # We want to find a solution with max < 1736 if possible.
    
    best_solution = [8, 28, 1736]
    best_max = 1736
    
    def is_valid(denoms):
        if not denoms or len(denoms) > 256:
            return False
        if len(set(denoms)) != len(denoms):
            return False
        if any(d < 1 or d > 10000000 for d in denoms):
            return False
        # Check sum exactly
        # Sum(1/d) = 5/31
        # To avoid Fraction overhead in tight loops, we can use integer arithmetic
        # but for final validation, Fraction is safe and fast enough for small lists.
        from fractions import Fraction
        s = sum(Fraction(1, d) for d in denoms)
        return s == Fraction(target_num, target_den)

    # Strategy:
    # We look for solutions of the form {a, b, c, d} or {a, b, c}.
    # The current best is {8, 28, 1736}.
    # Let's search for 3-term solutions first.
    # 1/a + 1/b + 1/c = 5/31
    # Assume a < b < c.
    # Since 1/a < 5/31, a > 31/5 = 6.2. So a >= 7.
    # Also 3/a >= 5/31 => a <= 93/5 = 18.6. So a <= 18.
    
    # For a given a, we need 1/b + 1/c = 5/31 - 1/a = (5a - 31) / (31a).
    # Let N = 5a - 31, D = 31a.
    # We need 1/b + 1/c = N/D.
    # This implies (b - D/N)(c - D/N) = (D/N)^2.
    # Or more simply, 1/b + 1/c = N/D => (b+c)/(bc) = N/D => Nbc = D(b+c) => Nbc - Db - Dc = 0.
    # Multiply by N: N^2 bc - DNb - DNe = 0.
    # (Nb - D)(Nc - D) = D^2.
    # Let X = Nb - D, Y = Nc - D.
    # Then XY = D^2.
    # b = (X + D) / N, c = (Y + D) / N.
    # For b and c to be integers, X must be a divisor of D^2 such that (X+D) is divisible by N.
    # Also b < c implies X < Y (since X = Nb - D).
    # And b > a.
    
    found_better = False
    
    # Iterate a from 7 to 18
    for a in range(7, 19):
        N = 5 * a - 31
        D = 31 * a
        
        if N <= 0:
            continue
            
        # We need to find divisors X of D^2 such that:
        # 1. X < D (so b < D/N + D/N = 2D/N? No, b = (X+D)/N. If X < D, b < 2D/N. 
        #    Actually, since b < c, X < sqrt(D^2) = D. So we only check X < D.
        #    Wait, if X = D, then b = 2D/N. If X > D, then b > c, which violates order.
        #    So we iterate X from 1 to sqrt(D^2) = D.
        #    However, D can be large. 31 * 18 = 558. D^2 is ~300k.
        #    Iterating all divisors of D^2 is efficient if we factor D.
        
        # Factor D
        def get_divisors(n):
            divs = set()
            i = 1
            while i * i <= n:
                if n % i == 0:
                    divs.add(i)
                    divs.add(n // i)
                i += 1
            return sorted(divs)
        
        # D is at most 558. D^2 is at most ~300,000.
        # Getting divisors of D^2 directly is fast.
        D_sq = D * D
        divs = get_divisors(D_sq)
        
        for X in divs:
            if X >= D: # We want b < c, so X < Y, and XY=D^2, so X < D.
                continue
                
            # Check if (X + D) is divisible by N
            if (X + D) % N != 0:
                continue
                
            b = (X + D) // N
            
            # b must be > a
            if b <= a:
                continue
                
            # Calculate c
            Y = D_sq // X
            if (Y + D) % N != 0:
                continue
            c = (Y + D) // N
            
            # c must be > b
            if c <= b:
                continue
                
            # Check distinctness (already ensured by a<b<c)
            # Check bounds
            if c > 10000000:
                continue
                
            # Validate the triple
            if is_valid([a, b, c]):
                curr_max = max(a, b, c)
                if curr_max < best_max:
                    best_max = curr_max
                    best_solution = [a, b, c]
                    found_better = True
                    
            # Early exit if we found a very small max?
            # If a=7, min b is 8. Min c?
            # 1/7 + 1/8 + 1/c = 5/31
            # 1/c = 5/31 - 1/7 - 1/8 = (280 - 248 - 217) / 1736 = -85/1736. Negative.
            # So 7 is not possible with b=8.
            # Let's just let the loop run, it's very fast.
            
    # If no 3-term solution better than 1736 was found, check 4-term solutions?
    # The current best is 1736.
    # Can we get max < 1736 with 4 terms?
    # 1/a + 1/b + 1/c + 1/d = 5/31.
    # If max < 1736, then d < 1736.
    # This search space is larger.
    # However, 3-term solutions are the most likely to have small max denominators.
    # If we didn't find a 3-term solution with max < 1736, it might be that 1736 is the minimum for 3 terms.
    # Let's quickly check if any 4-term solution exists with max < 1736.
    # We can use a similar divisor approach for the remainder after picking a, b.
    # 1/c + 1/d = R.
    # This is effectively the same as the 2-term completion used above.
    
    # Let's do a quick scan for 4-term solutions with a < b < c < d < 1736.
    # a >= 7.
    # If a=7, rem = 4/217.
    # 1/b + 1/c + 1/d = 4/217.
    # b > 7.
    # 3/b >= 4/217 => b <= 651/4 = 162.
    # So b is in [8, 162].
    # For each b, rem2 = 4/217 - 1/b = (4b - 217) / (217b).
    # Let N2 = 4b - 217, D2 = 217b.
    # We need 1/c + 1/d = N2/D2.
    # (c - D2/N2)(d - D2/N2) = (D2/N2)^2.
    # X Y = D2^2.
    # c = (X+D2)/N2, d = (Y+D2)/N2.
    
    # We want d < 1736.
    # This loop is small enough to run.
    
    for a in range(7, 19):
        if time.time() > deadline - 5.0:
            break
        N1 = 5 * a - 31
        D1 = 31 * a
        if N1 <= 0: continue
        
        # Max b for 3 remaining terms
        # 3/b >= N1/D1 => b <= 3*D1/N1
        max_b = int(3 * D1 / N1)
        # Min b is a+1
        min_b = a + 1
        
        for b in range(min_b, max_b + 1):
            if time.time() > deadline - 2.0:
                break
            N2 = 4 * b - 217 # Wait, remainder is N1/D1 - 1/b = (N1*b - D1) / (D1*b)
            # Let's re-calculate N2, D2 properly.
            # Rem = N1/D1 - 1/b = (N1*b - D1) / (D1*b)
            num2 = N1 * b - D1
            den2 = D1 * b
            
            if num2 <= 0:
                continue
                
            # We need 1/c + 1/d = num2/den2
            # c > b
            # 2/c >= num2/den2 => c <= 2*den2/num2
            max_c = int(2 * den2 / num2)
            min_c = b + 1
            
            # If max_c >= 1736, we might still find a solution with d < 1736?
            # No, if c < 1736 and d < 1736, then we are good.
            # But we are looking for ANY solution with max < 1736.
            # So we require d < 1736.
            # Since c < d, we also require c < 1736.
            
            # Iterate c?
            # c is in [min_c, min(max_c, 1735)]
            # For each c, check if d is integer and < 1736.
            # 1/d = num2/den2 - 1/c = (num2*c - den2) / (den2*c)
            # d = den2*c / (num2*c - den2)
            # Denominator must be positive: num2*c > den2 => c > den2/num2.
            # This is consistent with 1/c < num2/den2.
            
            start_c = max(min_c, int(den2 / num2) + 1)
            end_c = min(max_c, 1735)
            
            if start_c > end_c:
                continue
                
            for c in range(start_c, end_c + 1):
                den_part = num2 * c - den2
                if den_part <= 0:
                    continue
                num_part = den2 * c
                if num_part % den_part != 0:
                    continue
                d = num_part // den_part
                if d <= c:
                    continue
                if d >= 1736:
                    continue
                # Found candidate [a, b, c, d]
                if is_valid([a, b, c, d]):
                    curr_max = max(a, b, c, d)
                    if curr_max < best_max:
                        best_max = curr_max
                        best_solution = [a, b, c, d]
                        found_better = True

    # If we didn't find anything better, return the known best
    return best_solution
