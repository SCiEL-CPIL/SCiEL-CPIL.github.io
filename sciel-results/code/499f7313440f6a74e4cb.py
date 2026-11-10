"""Model-and-optimize: search for Egyptian fraction of 5/31 minimizing max denominator."""
import time

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s kill

    # Known solution: 1/8 + 1/31 + 1/248 = 5/31
    best_max = 248
    best_result = [8, 31, 248]

    def update_best(denoms):
        nonlocal best_max, best_result
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_result = list(denoms)

    # Search for 3-term solutions: 1/a + 1/b + 1/c = 5/31 with a < b < c < best_max
    # For each a: rem = 5/31 - 1/a = (5a-31)/(31a)
    # Need 1/b + 1/c = rem => rem*bc = b+c => (5a-31)bc = 31a(b+c)
    # => (5a-31)b(5a-31)c - 31a(5a-31)(b+c) = 0
    # Let X = (5a-31)b - 31a, Y = (5a-31)c - 31a
    # Then XY = (31a)^2
    # So (5a-31)b = X + 31a, so b = (X + 31a)/(5a-31)
    # X must divide (31a)^2, and (X + 31a) must be divisible by (5a-31)
    
    # a must satisfy: 1/a < 5/31 (so a > 31/5 = 6.2, so a >= 7)
    # and 3/a > 5/31 (so a < 93/5 = 18.6, so a <= 18)
    # Also a < best_max
    
    for a in range(7, min(19, best_max)):
        if time.time() > deadline:
            break
        num_rem = 5 * a - 31
        den_rem = 31 * a
        if num_rem <= 0:
            continue
        # (num_rem * b - den_rem) * (num_rem * c - den_rem) = den_rem^2
        # Let D = num_rem, N = den_rem
        # (D*b - N)(D*c - N) = N^2
        # Let X = D*b - N, so b = (X + N)/D, need D | (X+N)
        # X * Y = N^2, Y = N^2 / X
        # c = (Y + N)/D, need D | (Y+N)
        # Also b > a, c > b, c < best_max
        
        N_sq = den_rem * den_rem
        # X must be a divisor of N_sq
        # X > 0 (since b > a means D*b > D*a = 5a^2-31a, and N = 31a, 
        # D*a - N = 5a^2-31a-31a = 5a^2-62a = a(5a-62). For a>=13, this is positive.
        # For a < 13, D*a - N could be negative, but b > a so D*b - N > D*a - N.
        # We need X = D*b - N > 0 for b to be positive and reasonable.
        # Actually b > a => D*b > D*a => X = D*b - N > D*a - N.
        # D*a - N = 5a^2 - 31a - 31a = 5a^2 - 62a = a(5a-62)
        # For a=7: 7*(35-62) = 7*(-27) = -189. So X > -189, but we need X > 0 for b > N/D.
        # N/D = 31a/(5a-31). For a=7: 217/4 = 54.25. So b > 54.25, b >= 55.
        # X = D*b - N >= 4*55 - 217 = 220-217 = 3.
        
        # Iterate over divisors X of N_sq such that X > 0 and (X + N) % D == 0
        # X must be a divisor, so X <= N_sq. But b = (X+N)/D < best_max, so X < D*best_max - N.
        # Also b > a, so X > D*a - N.
        
        min_X = D * (a + 1) - N  # X > D*a - N, so X >= D*(a+1) - N
        if min_X < 1:
            min_X = 1
        max_X = D * (best_max - 1) - N  # X < D*best_max - N
        if max_X < min_X:
            continue
            
        # Find divisors of N_sq in [min_X, max_X]
        # N_sq can be large, so we iterate up to sqrt(N_sq)
        import math
        limit = int(math.isqrt(N_sq)) + 1
        for x in range(1, limit):
            if time.time() > deadline:
                break
            if N_sq % x == 0:
                for X in [x, N_sq // x]:
                    if X < min_X or X > max_X:
                        continue
                    if (X + den_rem) % num_rem != 0:
                        continue
                    b = (X + den_rem) // num_rem
                    if b <= a or b >= best_max:
                        continue
                    Y = N_sq // X
                    if (Y + den_rem) % num_rem != 0:
                        continue
                    c = (Y + den_rem) // num_rem
                    if c <= b or c >= best_max:
                        continue
                    # Verify: 1/a + 1/b + 1/c == 5/31
                    # num_rem/den_rem = 1/b + 1/c
                    # Check: num_rem * b * c == den_rem * (b + c)
                    if num_rem * b * c == den_rem * (b + c):
                        update_best([a, b, c])

    # Search for 4-term solutions: 1/a + 1/b + 1/c + 1/d = 5/31
    # For each a, b: rem = 5/31 - 1/a - 1/b
    # Then solve 1/c + 1/d = rem similarly
    # a: 7 to min(18, best_max-1) (need 4 terms, so 4/a > 5/31 => a < 124/5 = 24.8, but tighter)
    # Actually for 4 terms: 4/a >= 5/31 => a <= 124/5 = 24.8, so a <= 24
    # But also 1/a < 5/31 => a >= 7
    
    for a in range(7, min(25, best_max)):
        if time.time() > deadline:
            break
        num_rem_a = 5 * a - 31
        den_rem_a = 31 * a
        if num_rem_a <= 0:
            continue
        # b: 1/b < rem_a = num_rem_a/den_rem_a, so b > den_rem_a/num_rem_a
        # 3/b >= rem_a (since we need 3 more terms), so b <= 3*den_rem_a/num_rem_a
        # Also b > a, b < best_max
        min_b = max(a + 1, den_rem_a // num_rem_a + 1)
        max_b = min(3 * den_rem_a // num_rem_a, best_max - 1)
        
        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            if b >= best_max:
                break
            # rem2 = 5/31 - 1/a - 1/b
            # = (5ab - 31b - 31a) / (31ab)
            num2 = 5 * a * b - 31 * b - 31 * a
            den2 = 31 * a * b
            if num2 <= 0:
                continue
            # 1/c + 1/d = num2/den2
            # (num2*c - den2)(num2*d - den2) = den2^2
            # c > b, d > c, d < best_max
            min_c = max(b + 1, den2 // num2 + 1)
            max_c = min(2 * den2 // num2, best_max - 1)
            
            N2_sq = den2 * den2
            limit2 = int(math.isqrt(N2_sq)) + 1
            
            for c in range(min_c, max_c + 1):
                if time.time() > deadline:
                    break
                if c >= best_max:
                    break
                # d = den2 * c / (num2 * c - den2)
                # Need num2*c - den2 > 0 and den2*c % (num2*c - den2) == 0
                denom_d = num2 * c - den2
                if denom_d <= 0:
                    continue
                numer_d = den2 * c
                if numer_d % denom_d != 0:
                    continue
                d = numer_d // denom_d
                if d <= c or d >= best_max or d > 10000000:
                    continue
                # Verify
                if num2 * c * d == den2 * (c + d):
                    update_best([a, b, c, d])

    # Search for 5-term solutions
    for a in range(7, min(19, best_max)):
        if time.time() > deadline:
            break
        num_rem_a = 5 * a - 31
        den_rem_a = 31 * a
        if num_rem_a <= 0:
            continue
        min_b = max(a + 1, den_rem_a // num_rem_a + 1)
        max_b = min(4 * den_rem_a // num_rem_a, best_max - 1)
        
        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            if b >= best_max:
                break
            num2 = 5 * a * b - 31 * b - 31 * a
            den2 = 31 * a * b
            if num2 <= 0:
                continue
            min_c = max(b + 1, den2 // num2 + 1)
            max_c = min(3 * den2 // num2, best_max - 1)
            
            for c in range(min_c, max_c + 1):
                if time.time() > deadline:
                    break
                if c >= best_max:
                    break
                num3 = num2 * c - den2
                den3 = den2 * c
                if num3 <= 0:
                    continue
                min_d = max(c + 1, den3 // num3 + 1)
                max_d = min(2 * den3 // num3, best_max - 1)
                
                for d in range(min_d, max_d + 1):
                    if time.time() > deadline:
                        break
                    if d >= best_max:
                        break
                    # e = den3*d / (num3*d - den3)
                    denom_e = num3 * d - den3
                    if denom_e <= 0:
                        continue
                    numer_e = den3 * d
                    if numer_e % denom_e != 0:
                        continue
                    e = numer_e // denom_e
                    if e <= d or e >= best_max or e > 10000000:
                        continue
                    if num3 * d * e == den3 * (d + e):
                        update_best([a, b, c, d, e])

    if best_result is not None:
        return best_result
    return [8, 31, 248]

