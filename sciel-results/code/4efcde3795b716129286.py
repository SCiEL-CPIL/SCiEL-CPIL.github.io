"""Improved program for Egyptian fraction of 5/31; minimizes largest denominator."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    target = Fraction(5, 31)

    best_max = float('inf')
    best_solution = None

    def update_best(sol):
        nonlocal best_max, best_solution
        if sol and len(sol) <= 256:
            m = max(sol)
            if m < best_max:
                best_max = m
                best_solution = list(sol)

    # Strategy: Exhaustive search for 3-term solutions using divisor-based enumeration
    # For 1/a + 1/b + 1/c = 5/31 with a < b < c:
    # (a-b)(b-c)(c-a) relates to the equation, but more directly:
    # 31*(bc + ac + ab) = 5*abc
    # For fixed a, we need: 31*(b+c) + 31*bc/a = 5bc
    # => 31*(b+c) = bc*(5 - 31/a)
    # => 31*(b+c) = bc*(5a-31)/a
    # => 31a*(b+c) = bc*(5a-31)
    # Let k = 5a - 31. Then: 31a*(b+c) = k*bc
    # => k*bc - 31a*b - 31a*c = 0
    # => (kb - 31a)(kc - 31a) = (31a)^2
    # Let X = kb - 31a, Y = kc - 31a. Then XY = (31a)^2
    # So X and Y are positive divisors of (31a)^2 with X < Y (since b < c)
    # And b = (X + 31a)/k, c = (Y + 31a)/k must be integers > a.
    
    # We need k > 0, so 5a - 31 > 0, meaning a >= 7 (since a must be > 31/5 = 6.2)
    
    # Search for 3-term solutions with a < b < c, minimizing c
    found_3term = False
    
    for max_c in range(100, 3000, 10):
        if time.time() > deadline - 5.0:
            break
        if best_max <= max_c:
            break
            
        for a in range(7, max_c):
            if time.time() > deadline - 5.0:
                break
            k = 5 * a - 31
            if k <= 0:
                continue
            n = (31 * a) * (31 * a)
            
            # Find divisors X of n such that X < sqrt(n), Y = n/X > X
            # Then b = (X + 31a)/k, c = (Y + 31a)/k
            # Need b, c to be integers, b > a, c > b, c <= max_c
            
            # Iterate through divisors up to sqrt(n)
            import math
            sqrt_n = int(math.isqrt(n))
            
            for x in range(1, sqrt_n + 1):
                if time.time() > deadline - 5.0:
                    break
                if n % x != 0:
                    continue
                y = n // x
                # x < y (since x <= sqrt_n, y >= sqrt_n, and if x==y then b==c which we skip)
                if x >= y:
                    continue
                
                # b = (x + 31a) / k
                num_b = x + 31 * a
                if num_b % k != 0:
                    continue
                b = num_b // k
                if b <= a:
                    continue
                
                # c = (y + 31a) / k
                num_c = y + 31 * a
                if num_c % k != 0:
                    continue
                c = num_c // k
                if c <= b:
                    continue
                if c > max_c:
                    continue
                
                # Verify the solution
                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                    update_best([a, b, c])
                    found_3term = True
                    if best_max <= max_c:
                        break
            if best_max <= max_c:
                break
        
        if best_max <= max_c:
            break
    
    # If we found a 3-term solution, we can try to refine by searching lower bounds
    if found_3term and best_max < 3000:
        # Do a more precise search around the found value
        precise_max = best_max - 1
        if precise_max >= 100:
            for max_c in range(100, precise_max + 1, 1):
                if time.time() > deadline - 5.0:
                    break
                for a in range(7, max_c):
                    if time.time() > deadline - 5.0:
                        break
                    k = 5 * a - 31
                    if k <= 0:
                        continue
                    n = (31 * a) * (31 * a)
                    import math
                    sqrt_n = int(math.isqrt(n))
                    
                    for x in range(1, sqrt_n + 1):
                        if time.time() > deadline - 5.0:
                            break
                        if n % x != 0:
                            continue
                        y = n // x
                        if x >= y:
                            continue
                        num_b = x + 31 * a
                        if num_b % k != 0:
                            continue
                        b = num_b // k
                        if b <= a:
                            continue
                        num_c = y + 31 * a
                        if num_c % k != 0:
                            continue
                        c = num_c // k
                        if c <= b:
                            continue
                        if c > max_c:
                            continue
                        if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                            update_best([a, b, c])
                            break
                    if best_max <= max_c:
                        break
                if best_max <= max_c:
                    break
    
    # If no 3-term solution found or time remains, try 4-term solutions with bounded search
    if time.time() < deadline - 10.0 and (best_solution is None or best_max > 200):
        def dfs(r, min_den, max_den, current, depth, max_depth=256):
            nonlocal best_max, best_solution
            if time.time() > deadline - 2.0:
                return False
            if r == 0:
                if current:
                    m = max(current)
                    if m < best_max:
                        best_max = m
                        best_solution = list(current)
                return True
            if depth >= max_depth or r <= 0:
                return False
            
            d_start = max(min_den, (r.denominator + r.numerator - 1) // r.numerator)
            if d_start > max_den:
                return False
            
            # Pruning
            remaining_terms = max_depth - depth - 1
            if remaining_terms <= 0:
                return False
            if r > Fraction(remaining_terms, d_start + 1):
                return False
            
            # Try a limited number of candidates
            for d in range(d_start, min(d_start + 50, max_den + 1)):
                if time.time() > deadline - 2.0:
                    return False
                new_r = r - Fraction(1, d)
                if new_r < 0:
                    continue
                current.append(d)
                if dfs(new_r, d + 1, max_den, current, depth + 1, max_depth):
                    current.pop()
                    return True
                current.pop()
            
            return False
        
        # Try DFS with decreasing bounds
        search_upper = min(best_max - 1 if best_max < float('inf') else 500, 500)
        for search_max in range(search_upper, 100, -20):
            if time.time() > deadline - 10.0:
                break
            if dfs(target, 1, search_max, [], 0):
                if best_max <= search_max:
                    break
    
    # Verify and return
    if best_solution is not None and len(best_solution) <= 256:
        s = sum(Fraction(1, d) for d in best_solution)
        if s == target and len(set(best_solution)) == len(best_solution) and all(1 <= d <= 10000000 for d in best_solution):
            return best_solution
    
    # Fallback: greedy Egyptian fraction
    r = target
    values = []
    while r > 0:
        if time.time() > deadline - 1.0:
            break
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d <= 0 or d > 10000000:
            break
        if d in values:
            d += 1
            while d in values and d < 10000000:
                d += 1
            if d > 10000000:
                break
        values.append(d)
        r -= Fraction(1, d)
        if len(values) >= 256:
            break
    
    if values and sum(Fraction(1, d) for d in values) == target:
        return values
    
    # Absolute fallback
    return [8, 31, 248]
# EVOLVE-BLOCK-END
