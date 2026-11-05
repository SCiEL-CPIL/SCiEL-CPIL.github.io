import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    target = Fraction(5, 31)
    
    best_max = float('inf')
    best_solution = None
    
    # Exhaustive search for 3-term solutions 1/a + 1/b + 1/c = 5/31
    # with a < b < c
    # a must be in [7, 18]
    for a in range(7, 19):
        if time.time() > deadline:
            break
        # 1/b + 1/c = 5/31 - 1/a
        r = target - Fraction(1, a)
        if r <= 0:
            continue
        
        # (Nb - D)(Nc - D) = D^2 where N = r.denominator, D = r.numerator
        # Wait, r = N/D. 1/b + 1/c = N/D.
        # D(b+c) = Nbc
        # Nbc - Db - Dc = 0
        # (Nb - D)(Nc - D) = D^2
        N = r.denominator
        D = r.numerator
        
        # We need divisors X of D^2 such that X < D (for b < c)
        # and X > a*N - D (for a < b)
        # and (X+D) % N == 0
        # and (Y+D) % N == 0 where Y = D^2/X
        
        # To minimize max(a,b,c), we want to minimize c.
        # c = (Y+D)/N = (D^2/X + D)/N = D(D+X)/(NX)
        # This is minimized when X is maximized (subject to X < D).
        
        # Find max X < D such that X | D^2 and (X+D) % N == 0
        # We can iterate divisors of D^2
        # D is small (<= 31*18 = 558)
        # D^2 is <= 311364
        
        # Generate divisors of D^2
        divisors = set()
        for i in range(1, int(D**0.5) + 1):
            if D % i == 0:
                divisors.add(i)
                divisors.add(D // i)
        
        # D^2 divisors
        D2_divisors = set()
        for i in divisors:
            for j in divisors:
                D2_divisors.add(i * j)
        
        # Filter and find best
        for X in D2_divisors:
            if X >= D:
                continue
            if (X + D) % N != 0:
                continue
            Y = D2_divisors // X if D2_divisors % X == 0 else None # This is wrong, D2 is a number
            # D2 is D*D
            if (D * D) % X != 0:
                continue
            Y = (D * D) // X
            if (Y + D) % N != 0:
                continue
            
            b = (X + D) // N
            c = (Y + D) // N
            
            if b <= a or c <= b:
                continue
            
            if c < best_max:
                best_max = c
                best_solution = [a, b, c]
                
    # If no 3-term solution found (or time up), use the original logic
    if best_solution is None or time.time() > deadline - 5:
        # Fallback to original program logic
        def can_complete(r, min_den, max_den, max_terms=250):
            if r == 0:
                return []
            values = []
            used = set()
            for _ in range(max_terms):
                if time.time() > deadline:
                    return None
                if r == 0:
                    return values
                d_min = max(min_den, (r.denominator + r.numerator - 1) // r.numerator)
                if d_min > max_den:
                    return None
                d = d_min
                if d in used:
                    return None
                used.add(d)
                values.append(d)
                r -= Fraction(1, d)
                if r < 0:
                    return None
                min_den = d + 1
            return values if r == 0 else None
        
        for d1 in range(7, 60):
            if time.time() > deadline:
                break
            r1 = target - Fraction(1, d1)
            if r1 <= 0:
                continue
            d2_min = (r1.denominator + r1.numerator - 1) // r1.numerator
            d2_min = max(d2_min, d1 + 1)
            for d2 in range(d2_min, min(d2_min + 50, 500)):
                if time.time() > deadline:
                    break
                r2 = r1 - Fraction(1, d2)
                if r2 <= 0:
                    continue
                sol = can_complete(r2, d2 + 1, best_max - 1 if best_max < float('inf') else 500)
                if sol is not None:
                    full_sol = [d1, d2] + sol
                    if len(full_sol) <= 256:
                        m = max(full_sol)
                        if m < best_max:
                            best_max = m
                            best_solution = full_sol

    if best_solution is not None:
        return best_solution
    else:
        # Absolute fallback
        return [8, 31, 248]
