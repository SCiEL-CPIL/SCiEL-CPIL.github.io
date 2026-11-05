def construct():
    from fractions import Fraction
    target = Fraction(5, 31)
    best = None
    best_max = float('inf')
    
    # Search for 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # Using the constraint: 5(5ab - 31a - 31b) | 31(a+b)
    # Which implies n = 31(a+b) / (5ab - 31a - 31b) is an integer
    # And (5a-31)(5b-31) = 961 * (n+1)/n ? No, derived as:
    # (lambda a - 31)(lambda b - 31) = 961 where lambda = 5n/(n+1)
    
    # We iterate n such that solutions exist.
    # From (u+31)(n+1) / 5n = a, we found n must divide A_u and satisfy mod 5 condition.
    # We can just iterate n up to a reasonable bound and check divisors of 961.
    
    # Divisors of 961
    divs = [1, 31, 961]
    
    # n can be large?
    # a = (u+31)(n+1)/5n.
    # If n is large, a approx (u+31)/5.
    # We need a to be integer.
    # We can iterate n from 1 to some limit, say 1000.
    # Or iterate divisors of A_u as before.
    
    # Let's just iterate n from 1 to 10000 to be safe, and check if a,b are integers.
    # This is O(N * 3) which is fast.
    
    for n in range(1, 10001):
        for u in divs:
            v = 961 // u
            # a = (u+31)(n+1) / 5n
            num_a = (u + 31) * (n + 1)
            den_a = 5 * n
            if num_a % den_a != 0:
                continue
            a = num_a // den_a
            if a < 1:
                continue
            
            num_b = (v + 31) * (n + 1)
            den_b = 5 * n
            if num_b % den_b != 0:
                continue
            b = num_b // den_b
            if b < 1:
                continue
            
            if a == b:
                continue
                
            # Calculate c
            # c = n * a * b / (a + b)
            num_c = n * a * b
            den_c = a + b
            if num_c % den_c != 0:
                continue
            c = num_c // den_c
            if c < 1:
                continue
            
            if c == a or c == b:
                continue
            
            # Verify
            s = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
            if s != target:
                continue
                
            m = max(a, b, c)
            if m < best_max:
                best_max = m
                best = [a, b, c]
                
    if best is not None:
        return best
        
    # Fallback
    r, values = Fraction(5, 31) - Fraction(1, 8), [8]
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
