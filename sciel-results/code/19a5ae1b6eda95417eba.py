def construct():
    import time
    from fractions import Fraction
    import math
    
    deadline = time.time() + 75
    target = Fraction(5, 31)
    best_solution = None
    best_max = float('inf')
    
    # Search 3-term: 5/31 = 1/a + 1/b + 1/c
    # For each a >= 7, compute remainder = 5/31 - 1/a = (5a-31)/(31a)
    # Solve 1/b + 1/c = (5a-31)/(31a) via (31a*b - (5a-31))(31a*c - (5a-31)) = (5a-31)^2
    
    for a in range(7, 100):
        if time.time() > deadline:
            break
        num = 5 * a - 31
        if num <= 0:
            continue
        den = 31 * a
        k2 = num * num
        sqrt_k2 = int(math.isqrt(k2))
        
        for u in range(1, sqrt_k2 + 1):
            if k2 % u != 0:
                continue
            v = k2 // u
            bu = u + num
            bv = v + num
            if bu % den == 0 and bv % den == 0:
                b = bu // den
                c = bv // den
                if b > 0 and c > 0 and b != c and b != a and c != a:
                    mx = max(a, b, c)
                    if mx < best_max:
                        best_max = mx
                        best_solution = sorted([a, b, c])
    
    if best_solution:
        s = sum(Fraction(1, x) for x in best_solution)
        if s == target:
            return best_solution
    
    # Greedy fallback
    r = target
    vals = []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        vals.append(d)
        r -= Fraction(1, d)
    return vals
