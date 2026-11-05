def construct():
    from fractions import Fraction
    import math

    target = Fraction(5, 31)
    
    best_max = float('inf')
    best_solution = None
    
    # Helper to find c, d for 1/c + 1/d = r, with c > b_min and d > c, and d < max_limit
    def find_cd(r, b_min, max_limit):
        p = r.numerator
        q = r.denominator
        q2 = q * q
        limit = int(math.isqrt(q2))
        
        # Generate divisors of q2
        # For efficiency, we can just iterate up to sqrt
        # But q2 can be large. 
        # However, we only care about solutions where c, d < max_limit.
        # c > b_min.
        # 1/c + 1/d = p/q => d = q*c / (p*c - q)
        # d > c => q*c / (p*c - q) > c => q / (p*c - q) > 1 => q > p*c - q => 2q > p*c => c < 2q/p
        # Also d < max_limit => q*c / (p*c - q) < max_limit => q*c < max_limit*(p*c - q) => q*c < max_limit*p*c - max_limit*q => max_limit*q < c*(max_limit*p - q) => c > max_limit*q / (max_limit*p - q)
        
        # So c is in range (max(b_min, max_limit*q / (max_limit*p - q)), 2q/p)
        
        lower_bound = b_min + 1
        upper_bound = int(2 * q / p)
        
        # Refine lower bound from max_limit constraint
        # If max_limit*p - q <= 0, then p*c - q >= p*lower_bound - q.
        # If p*lower_bound <= q, then 1/c + 1/d <= 1/c + 1/(c+1) < 2/c.
        # If 2/c <= p/q, no solution.
        
        # Let's just iterate c from lower_bound to upper_bound
        # and check if d is integer and valid.
        # This might be faster than factorization if the range is small.
        # Range size: 2q/p - b_min.
        
        for c in range(lower_bound, upper_bound + 1):
            num = p * c - q
            if num <= 0:
                continue
            if (q * c) % num != 0:
                continue
            d = (q * c) // num
            if d <= c:
                continue
            if d >= max_limit:
                continue
            # Check distinctness with previous terms? 
            # Caller ensures c > b_min, so c > b.
            # We assume a, b are distinct and < c.
            return [c, d]
        return None

    # 3-term search
    for a in range(7, 100):
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            continue
        
        # Find b, c such that 1/b + 1/c = r1, b > a, c > b
        # Use factorization method as in original, or iteration
        # Iteration is simpler to bound by max_limit
        p = r1.numerator
        q = r1.denominator
        
        # b < 2q/p
        b_upper = int(2 * q / p)
        
        for b in range(a + 1, b_upper + 1):
            num = p * b - q
            if num <= 0:
                continue
            if (q * b) % num != 0:
                continue
            c = (q * b) // num
            if c <= b:
                continue
            max_den = c
            if max_den < best_max:
                best_max = max_den
                best_solution = [a, b, c]
                
        # Early break if best_max is very small?
        if best_solution is not None and best_max < a * 2:
            break

    # 4-term search
    # Look for a < b < c < d < best_max
    # If we find one, it's better.
    # We only need to search if best_max is not 1 (impossible) or if we can beat it.
    # Since best_max is likely 248, we search for d < 248.
    
    if best_solution is not None:
        limit_d = best_max
        for a in range(7, limit_d):
            r1 = target - Fraction(1, a)
            if r1 <= 0:
                continue
            
            # b > a
            # 1/b < r1 => b > 1/r1
            # 1/b > r1 - 2/(b+1) ... 
            # Just bound b by 1/r1 roughly? No, 1/b < r1 is necessary.
            # Also 1/b + 1/c + 1/d = r1 => 3/d < r1 => d > 3/r1.
            # Since d < limit_d, we need 3/r1 < limit_d => r1 > 3/limit_d.
            if r1 <= Fraction(3, limit_d):
                continue
                
            b_upper = int(1 / r1) # b must be > 1/r1? No, 1/b < r1 => b > 1/r1.
            # Actually 1/b < r1 is not sufficient, we need 1/b + 1/c + 1/d = r1.
            # 1/b < r1 => b > 1/r1.
            
            for b in range(a + 1, b_upper + 1):
                r2 = r1 - Fraction(1, b)
                if r2 <= 0:
                    continue
                # Check if r2 can be sum of 2 reciprocals < limit_d
                # 2/d < r2 => d > 2/r2
                if r2 <= Fraction(2, limit_d):
                    continue
                
                res_cd = find_cd(r2, b, limit_d)
                if res_cd:
                    c, d = res_cd
                    max_den = d
                    if max_den < best_max:
                        best_max = max_den
                        best_solution = [a, b, c, d]
                        
            # Prune a if no b works?
            # If a is large, r1 is small.
            
    if best_solution is None:
        # Fallback
        r = target
        values = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values

    return best_solution
