def construct():
    from fractions import Fraction
    import math

    target = Fraction(5, 31)
    best_max = float('inf')
    best_solution = None

    # We will search for solutions with 3, 4, and 5 terms.
    # For a given number of terms k, we want to minimize the largest denominator.
    # A solution with fewer terms generally has smaller denominators, but we must check all.
    # However, the problem asks to minimize the LARGEST denominator.
    # It is possible that a 4-term solution has a smaller max denominator than any 3-term solution.
    # But typically, fewer terms allow for larger individual denominators to be avoided? No.
    # 1/a + 1/b + 1/c = 5/31. If we split 1/c into 1/d + 1/e, then 1/d + 1/e = 1/c.
    # This implies d, e > c. So the max denominator increases.
    # Therefore, the optimal solution should have the minimum number of terms possible.
    # We established that 2 terms is impossible. So we check 3 terms first.
    # If a 3-term solution exists, its max denominator is likely smaller than any 4-term solution's max denominator.
    # Why? Let S3 be a 3-term solution with max C.
    # Any 4-term solution derived from it has max > C.
    # Could there be a "disjoint" 4-term solution with max < C?
    # Let the 4 terms be a<b<c<d.
    # If d < C, then 1/a+1/b+1/c+1/d = 5/31.
    # This is a distinct search space.
    # However, the greedy method or factorization for 3 terms is very fast.
    # Let's first find the best 3-term solution.
    # Then, we can check if a 4-term solution can beat it.
    # Given the constraints (60s), we can afford a slightly deeper search for 4 terms if needed,
    # but 3 terms is the primary candidate for the optimum.

    def find_solution(k_terms):
        nonlocal best_max, best_solution
        
        if k_terms == 3:
            # 1/a + 1/b + 1/c = 5/31, a < b < c
            # a >= 7
            # 1/a < 5/31 => a > 6.2
            # Also 3/a > 5/31 => a < 3*31/5 = 18.6 => a <= 18
            # So a is in [7, 18]
            for a in range(7, 19):
                r1 = target - Fraction(1, a)
                if r1 <= 0:
                    continue
                # 1/b + 1/c = r1 = p/q
                p = r1.numerator
                q = r1.denominator
                q2 = q * q
                # (pb-q)(pc-q) = q^2
                # d1 = pb-q, d2 = pc-q
                # d1*d2 = q^2
                # b = (d1+q)/p, c = (d2+q)/p
                # We need d1, d2 to be divisors of q^2.
                # And b > a, c > b.
                limit = int(math.isqrt(q2))
                for d1 in range(1, limit + 1):
                    if q2 % d1 != 0:
                        continue
                    d2 = q2 // d1
                    
                    # Case 1: d1 <= d2 => b <= c
                    if (d1 + q) % p == 0 and (d2 + q) % p == 0:
                        b = (d1 + q) // p
                        c = (d2 + q) // p
                        if b > a and c > b:
                            if c < best_max:
                                best_max = c
                                best_solution = [a, b, c]
                    
                    # Case 2: d2 <= d1 => c <= b (swapped)
                    # Since we enforce a<b<c, we just need to check if the other assignment works
                    # But d1, d2 are just divisors. The set {d1, d2} is the same.
                    # The values b, c are determined by which divisor is assigned to b.
                    # If we swap d1 and d2, b and c swap.
                    # Since we require b < c, we only care about the case where the resulting b < c.
                    # In Case 1, if d1 < d2, then b < c. If d1 > d2, then b > c (invalid for this assignment).
                    # If d1 == d2, b == c (invalid).
                    # So we just need to check if b < c.
                    # Wait, my loop iterates d1 up to sqrt. So d1 <= d2 is guaranteed.
                    # Thus b <= c.
                    # We just need b < c (distinct).
        
        elif k_terms == 4:
            # 1/a + 1/b + 1/c + 1/d = 5/31, a < b < c < d
            # a >= 7
            # 4/a > 5/31 => a < 4*31/5 = 24.8 => a <= 24
            # If best_max is already found from 3 terms, we can prune.
            # d < best_max
            # Also 1/d > 0.
            # Let's bound a, b, c.
            # This search space is larger.
            # We only do this if we want to be sure.
            # Given 3 terms likely gives ~250, and 4 terms will have max > 250 usually.
            # Let's check if a 4-term solution can have max < current best_max.
            # If best_max is inf, we must do it.
            # If best_max is ~250, can we find 4 terms with max < 250?
            # 1/a+1/b+1/c+1/d = 5/31.
            # If d < 250, then 1/d > 1/250.
            # This is a valid search.
            
            if best_max == float('inf'):
                max_a = 24
            else:
                # d < best_max
                # 1/a + 1/b + 1/c > 5/31 - 1/best_max
                # This doesn't bound a tightly enough for a simple loop without deeper recursion.
                # Let's just iterate a, b, c and solve for d.
                # 1/d = 5/31 - 1/a - 1/b - 1/c
                # d = 1 / (5/31 - 1/a - 1/b - 1/c)
                # d must be integer > c and d < best_max.
                
                max_a = 24
            
            for a in range(7, max_a + 1):
                r1 = target - Fraction(1, a)
                if r1 <= 0: continue
                # 1/b + 1/c + 1/d = r1
                # b > a
                # 3/b > r1 => b < 3/r1
                max_b = int(3 / float(r1)) + 1
                # Also b < best_max (since d > b)
                if best_max != float('inf'):
                    max_b = min(max_b, best_max - 1)
                
                for b in range(a + 1, max_b + 1):
                    r2 = r1 - Fraction(1, b)
                    if r2 <= 0: continue
                    # 1/c + 1/d = r2
                    # c > b
                    # 2/c > r2 => c < 2/r2
                    max_c = int(2 / float(r2)) + 1
                    if best_max != float('inf'):
                        max_c = min(max_c, best_max - 1)
                    
                    for c in range(b + 1, max_c + 1):
                        r3 = r2 - Fraction(1, c)
                        if r3 <= 0: continue
                        # 1/d = r3 => d = 1/r3
                        if r3.numerator == 1:
                            d = r3.denominator
                            if d > c:
                                if best_max != float('inf') and d >= best_max:
                                    continue
                                if d < best_max:
                                    best_max = d
                                    best_solution = [a, b, c, d]
                                elif best_max == float('inf'):
                                    best_max = d
                                    best_solution = [a, b, c, d]

    # Start with 3 terms
    find_solution(3)
    
    # If we found a 3-term solution, check if 4-term can improve it.
    # It's unlikely but possible.
    # If best_max is small, the search space for 4 terms is pruned heavily.
    if best_solution is not None:
        find_solution(4)
    else:
        # If no 3-term solution found (shouldn't happen), search 4 terms without pruning
        find_solution(4)
        # If still none, try 5 terms? Unlikely needed.
        # Fallback to greedy if all else fails (should not happen)
        if best_solution is None:
            r = target
            values = []
            while r:
                d = (r.denominator + r.numerator - 1) // r.numerator
                values.append(d)
                r -= Fraction(1, d)
            return values

    return best_solution
