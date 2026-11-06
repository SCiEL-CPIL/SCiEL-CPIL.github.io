"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from fractions import Fraction
    
    target = Fraction(5, 31)
    deadline = time.time() + 85
    
    best_max = None
    best_solution = None
    
    def try_update(solution):
        nonlocal best_max, best_solution
        m = max(solution)
        if best_max is None or m < best_max:
            best_max = m
            best_solution = list(solution)
    
    # Try 2 terms: 1/a + 1/b = 5/31
    # (a+b)/(ab) = 5/31 => 31(a+b) = 5ab => 5ab - 31a - 31b = 0
    # (5a - 31)(5b - 31) = 31^2 = 961
    # Factor pairs of 961: 1*961, 31*31
    # 5a-31=1, 5b-31=961 => a=32/5 not integer
    # 5a-31=31, 5b-31=31 => a=62/5 not integer
    # So no 2-term solution.
    
    # Try 3 terms: 1/a + 1/b + 1/c = 5/31, a < b < c
    # For each a from 2 upward, solve for b, c
    # 1/b + 1/c = 5/31 - 1/a = (5a - 31)/(31a)
    # Let r = (5a - 31)/(31a), need r > 0 so a > 31/5 = 6.2, so a >= 7
    # (b+c)/(bc) = r => r*bc = b+c => (rb-1)(rc-1) = 1
    # Actually: rbc - b - c = 0 => (rb-1)(rc-1) = 1 + r
    # Let me use: 1/b + 1/c = r => c = b/(rb - 1)
    # So for each b, c = b/(rb - 1) must be integer > b
    
    # a must satisfy: 1/a < 5/31, so a > 31/5 = 6.2, a >= 7
    # Also 1/a + 1/b + 1/c = 5/31, with b > a, c > b
    # So 3/a > 5/31 (since each term <= 1/a) => a < 93/5 = 18.6, so a <= 18
    
    for a in range(7, 19):
        if time.time() > deadline:
            break
        r = target - Fraction(1, a)
        if r <= 0:
            continue
        # Now find b > a such that c = b/(r*b - 1) is integer > b
        # r*b - 1 must divide b, and c > b means b/(r*b-1) > b => 1/(r*b-1) > 1 => r*b-1 < 1 => r*b < 2
        # Wait, that's not right. c = b/(r*b - 1), and we need c > b
        # b/(r*b-1) > b => 1/(r*b-1) > 1 => r*b-1 < 1 => r*b < 2
        # But r = 5/31 - 1/a, for a=7, r = 5/31-1/7 = (35-31)/217 = 4/217
        # r*b < 2 => b < 2/r = 2*217/4 = 108.5, so b <= 108
        # Also r*b - 1 > 0 => b > 1/r = 217/4 = 54.25, so b >= 55
        # And b > a = 7, which is satisfied
        
        # c = b/(r*b - 1) = b / ((5a-31)/(31a) * b - 1) = b / ((5a-31)*b - 31a)/(31a) = 31a*b / ((5a-31)*b - 31a)
        # So (5a-31)*b - 31a must divide 31a*b
        # Let g = gcd(31a, (5a-31)*b - 31a), but this is complex
        
        # Simpler: iterate b and check if c is integer
        # b range: b > a and r*b > 1 (so c > 0) and c > b
        # c > b => 31a*b / ((5a-31)*b - 31a) > b => 31a / ((5a-31)*b - 31a) > 1 => 31a > (5a-31)*b - 31a => 62a > (5a-31)*b => b < 62a/(5a-31)
        # Also r*b > 1 => b > 1/r = 31a/(5a-31)
        # And b > a
        
        b_min = max(a + 1, 31 * a // (5 * a - 31) + 1)
        b_max = min(62 * a // (5 * a - 31), 10000000)  # cap at max allowed
        
        if b_min > b_max:
            continue
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            # c = 31*a*b / ((5*a-31)*b - 31*a)
            num = 31 * a * b
            den = (5 * a - 31) * b - 31 * a
            if den <= 0:
                continue
            if num % den != 0:
                continue
            c = num // den
            if c <= b:
                continue
            if c > 10000000:
                continue
            # Verify
            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                try_update([a, b, c])
    
    # Try 4 terms if needed
    if best_max is None or best_max > 50:
        # 1/a + 1/b + 1/c + 1/d = 5/31
        # a < b < c < d
        # a >= 7 (as before), a <= ? 4/a > 5/31 => a < 124/5 = 24.8, a <= 24
        for a in range(7, 25):
            if time.time() > deadline:
                break
            r1 = target - Fraction(1, a)
            if r1 <= 0:
                continue
            # 1/b + 1/c + 1/d = r1
            # b > a, b <= ? 3/b > r1 => b < 3/r1
            # Also 1/b < r1 => b > 1/r1
            b_min = a + 1
            b_max = min(int(3 / r1) + 1, 10000000)
            if b_min > b_max:
                continue
            for b in range(b_min, b_max + 1):
                if time.time() > deadline:
                    break
                r2 = r1 - Fraction(1, b)
                if r2 <= 0:
                    continue
                # 1/c + 1/d = r2
                # c > b, c <= ? 2/c > r2 => c < 2/r2
                # Also 1/c < r2 => c > 1/r2
                c_min = b + 1
                c_max = min(int(2 / r2) + 1, 10000000)
                if c_min > c_max:
                    continue
                for c in range(c_min, c_max + 1):
                    if time.time() > deadline:
                        break
                    r3 = r2 - Fraction(1, c)
                    if r3 <= 0:
                        continue
                    # 1/d = r3 => d = 1/r3
                    if r3.denominator == 1:
                        d = r3.numerator
                        if d > c and d <= 10000000:
                            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == target:
                                try_update([a, b, c, d])
                    else:
                        # d must be integer
                        if r3.numerator == 1:
                            d = r3.denominator
                            if d > c and d <= 10000000:
                                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == target:
                                    try_update([a, b, c, d])
    
    # Fallback to greedy
    if best_solution is None:
        r = target
        values = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        best_solution = values
    
    return best_solution
# EVOLVE-BLOCK-END
