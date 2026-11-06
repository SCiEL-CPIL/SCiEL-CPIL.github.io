def construct():
    import time
    from fractions import Fraction
    import math
    
    target = Fraction(5, 31)
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    best_max = float('inf')
    best_denoms = []
    
    # Greedy initialization
    def greedy_solve(r, used):
        denoms = []
        r = Fraction(r)
        while r > 0:
            k_min = (r.denominator + r.numerator - 1) // r.numerator
            k = k_min
            while k in used:
                k += 1
            denoms.append(k)
            used.add(k)
            r -= Fraction(1, k)
        return denoms
    
    # Quick greedy to get initial bound
    initial_greedy = greedy_solve(target, set())
    best_max = max(initial_greedy)
    best_denoms = initial_greedy[:]
    
    # Helper: find the last two denominators given a remainder r = a/b
    # We need 1/x + 1/y = a/b with x, y distinct and not in used
    # => b(y+x) = axy
    # => axy - by - bx = 0
    # => (ax - b)(ay - b) = b^2
    # So ax - b must be a divisor of b^2
    # Let d = ax - b, then x = (d + b) / a
    # And ay - b = b^2 / d, so y = (b^2/d + b) / a
    
    def find_two_terms(r, used, max_bound):
        # r = a/b in lowest terms
        a = r.numerator
        b = r.denominator
        b2 = b * b
        results = []
        
        # Enumerate divisors of b^2
        # d must satisfy: d | b^2, d > 0, (d+b) % a == 0, x = (d+b)//a
        # and e = b^2//d, (e+b) % a == 0, y = (e+b)//a
        # x != y, x not in used, y not in used, x <= max_bound, y <= max_bound
        # Also need 1/x + 1/y = a/b exactly
        
        # Factorize b to enumerate divisors of b^2
        # Actually, let's just iterate d up to sqrt(b^2) = b
        # But b can be large. Let's be smarter.
        # d must be a divisor of b^2.
        # Number of divisors of b^2 can be large but manageable for b up to ~10^6
        # Actually b can grow. Let's use a different approach.
        
        # Alternative: x must satisfy that (ax - b) divides b^2
        # And x >= ceil(b/a) (since 1/x <= a/b => x >= b/a)
        # Also 1/x + 1/y = a/b => 1/y = a/b - 1/x = (ax - b)/(bx)
        # So y = bx / (ax - b)
        # For y to be integer, (ax - b) must divide bx
        # Since gcd(ax-b, x) = gcd(-b, x) = gcd(b, x)
        # This is getting complicated. Let's stick with divisor enumeration of b^2.
        
        # Find all divisors of b^2
        def divisors(n):
            if n == 1:
                return [1]
            result = set()
            i = 1
            while i * i <= n:
                if n % i == 0:
                    result.add(i)
                    result.add(n // i)
                i += 1
            return sorted(result)
        
        divs = divisors(b2)
        
        for d in divs:
            # d = ax - b => x = (d + b) / a
            num_x = d + b
            if num_x % a != 0:
                continue
            x = num_x // a
            if x <= 0:
                continue
            if x in used:
                continue
            if x > max_bound:
                continue
            
            e = b2 // d
            num_y = e + b
            if num_y % a != 0:
                continue
            y = num_y // a
            if y <= 0:
                continue
            if y in used:
                continue
            if y > max_bound:
                continue
            if x == y:
                continue
            
            # Verify
            if Fraction(1, x) + Fraction(1, y) == r:
                mx = max(x, y)
                if mx < max_bound:
                    results.append((mx, x, y))
        
        results.sort()
        return results
    
    # Search with up to 4 terms: fix first 2, solve last 2 algebraically
    # d1 > 6.2, so d1 >= 7
    # d2 > d1, and 1/d2 < target - 1/d1
    
    found_better = False
    
    for d1 in range(7, 1000):
        if time.time() > deadline - 5:
            break
        if d1 >= best_max:
            break
            
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        
        used = {d1}
        
        # d2 must satisfy 1/d2 < r1 => d2 > 1/r1
        # Also d2 > d1
        d2_min = max(d1 + 1, (r1.denominator + r1.numerator - 1) // r1.numerator)
        
        # Upper bound on d2: if d2 is too large, then r2 = r1 - 1/d2 is close to r1
        # and the last two terms would need to sum to ~r1, requiring large denominators
        # Let's bound d2 such that max(d3, d4) < best_max
        # 1/d2 + 1/d3 + 1/d4 = r1
        # If d3, d4 < best_max, then 1/d3 + 1/d4 > 2/best_max
        # So 1/d2 < r1 - 2/best_max (roughly)
        # Actually: 1/d2 = r1 - (1/d3 + 1/d4) > r1 - 2/(d3) 
        # This is getting complex. Let's just iterate d2 up to a reasonable limit
        # and use the algebraic solver for the last two.
        
        # d2 must also be such that r1 - 1/d2 > 0 => d2 > 1/r1
        # And 1/d2 + 1/d3 + 1/d4 = r1 with d3,d4 < best_max
        # Max of 1/d3 + 1/d4 when d3,d4 < best_max is approximately 2/(d2+1) if d3,d4 just above d2
        # This is a loose bound. Let's iterate d2 up to min(2*best_max, 50000)
        
        d2_limit = min(2 * best_max, 50000)
        
        for d2 in range(d2_min, d2_limit + 1):
            if time.time() > deadline - 5:
                break
            if d2 >= best_max:
                break
                
            r2 = r1 - Fraction(1, d2)
            if r2 <= 0:
                continue
            
            used2 = used | {d2}
            
            # Find d3, d4 such that 1/d3 + 1/d4 = r2
            two_terms = find_two_terms(r2, used2, best_max)
            
            if two_terms:
                mx, d3, d4 = two_terms[0]
                current_max = max(d1, d2, d3, d4)
                if current_max < best_max:
                    best_max = current_max
                    best_denoms = [d1, d2, d3, d4]
                    found_better = True
                    # Update bounds
                    d2_limit = min(2 * best_max, 50000)
                    
    # Also try 3-term solutions: fix first term, solve last two
    for d1 in range(7, best_max):
        if time.time() > deadline - 2:
            break
            
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        
        used = {d1}
        two_terms = find_two_terms(r1, used, best_max)
        
        if two_terms:
            mx, d2, d3 = two_terms[0]
            current_max = max(d1, d2, d3)
            if current_max < best_max:
                best_max = current_max
                best_denoms = [d1, d2, d3]
    
    # Also try 2-term solutions
    for d1 in range(7, best_max):
        if time.time() > deadline - 1:
            break
            
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        
        used = {d1}
        # 1/d2 = r1 => d2 = 1/r1
        if r1.numerator != 1:
            continue
        d2 = r1.denominator
        if d2 in used or d2 >= best_max:
            continue
        if Fraction(1, d1) + Fraction(1, d2) == target:
            current_max = max(d1, d2)
            if current_max < best_max:
                best_max = current_max
                best_denoms = [d1, d2]
    
    # If we found something better, return it; otherwise return the greedy solution
    if best_denoms:
        return best_denoms
    else:
        return initial_greedy
