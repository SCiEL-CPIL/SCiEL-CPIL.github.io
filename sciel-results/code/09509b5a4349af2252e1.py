import time
from fractions import Fraction

def construct():
    start_time = time.time()
    target = Fraction(5, 31)
    
    # Known best so far
    best_sol = [8, 31, 248]
    best_max = 248
    
    def check_solution(sol):
        if not sol:
            return False
        if len(sol) > 256:
            return False
        if len(set(sol)) != len(sol):
            return False
        if any(d < 1 or d > 10000000 for d in sol):
            return False
        s = sum(Fraction(1, d) for d in sol)
        return s == target

    # Search for 3-term solutions
    # 1/a + 1/b + 1/c = 5/31
    # a < b < c
    # a >= 7
    # c = 31ab / (5ab - 31a - 31b)
    # We want c < best_max
    # Iterate a
    for a in range(7, 100):
        if time.time() - start_time > 50:
            break
        # b > 31a / (5a - 31)
        min_b = int(31 * a / (5 * a - 31)) + 1
        # c > b
        # c = 31ab / D
        # D = 5ab - 31a - 31b
        # We need D > 0
        # 5ab > 31a + 31b
        # b(5a - 31) > 31a
        # b > 31a / (5a - 31)
        # Also c < best_max
        # 31ab / D < best_max
        # 31ab < best_max * D = best_max * (5ab - 31a - 31b)
        # 31ab < 5 * best_max * ab - 31 * best_max * a - 31 * best_max * b
        # ab (5 * best_max - 31) > 31 * best_max * (a + b)
        # This gives an upper bound on b for a given a if we want small c.
        # But we just want ANY c < best_max.
        # Let's just iterate b.
        # b must be such that c is integer and c > b and c < best_max.
        # c < best_max implies D > 31ab / best_max.
        # 5ab - 31a - 31b > 31ab / best_max
        # ab (5 - 31/best_max) > 31(a+b)
        
        # Upper bound for b:
        # As b increases, c decreases.
        # If c <= b, we stop (since we require c > b).
        # c = 31ab / (5ab - 31a - 31b)
        # c <= b => 31a <= 5ab - 31a - 31b => 62a + 31b <= 5ab => b(5a - 31) >= 62a => b >= 62a / (5a - 31)
        # So we only need to check b up to floor(62a / (5a - 31)) - 1?
        # No, if b is large, c is small. We want c > b.
        # So we stop when c <= b.
        
        max_b = 200 # Safe limit for 3 terms to find small c
        for b in range(min_b, max_b):
            if time.time() - start_time > 55:
                break
            D = 5 * a * b - 31 * a - 31 * b
            if D <= 0:
                continue
            if (31 * a * b) % D != 0:
                continue
            c = (31 * a * b) // D
            if c <= b:
                break # c decreases as b increases, so further b will have smaller c
            if c < best_max:
                sol = [a, b, c]
                if check_solution(sol):
                    return sol
                    # Update best if we want to find even smaller? 
                    # The task is to minimize largest denominator.
                    # If we find one, we can return it immediately if we search in order of max denom?
                    # No, we are iterating a, b. We don't guarantee c is minimal.
                    # But we can keep the best found.
                    best_sol = sol
                    best_max = c
                    
    # Search for 4-term solutions
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # a < b < c < d < best_max
    # d = 31 / (5 - 31(1/a+1/b+1/c))
    # Let S3 = 1/a + 1/b + 1/c
    # d = 31 / (5 - 31 S3)
    # Numerator 31, Denominator 5 - 31S3.
    # 5 - 31S3 must divide 31.
    # Since S3 > 0, 5 - 31S3 < 5.
    # Also S3 < 5/31, so 5 - 31S3 > 0.
    # So 5 - 31S3 is a positive rational number less than 5.
    # For d to be integer, 5 - 31S3 must be a divisor of 31?
    # No, d = 31 / X. X must be a divisor of 31 for d to be integer?
    # Yes, if X is rational p/q, d = 31q/p.
    # Let's use Fraction.
    
    if time.time() - start_time < 55:
        # a from 7
        for a in range(7, 50):
            if time.time() - start_time > 58:
                break
            # b from a+1
            for b in range(a + 1, 100):
                if time.time() - start_time > 59:
                    break
                # c from b+1
                # 1/a + 1/b + 1/c < 5/31
                # c > 1 / (5/31 - 1/a - 1/b)
                rem_ab = target - Fraction(1, a) - Fraction(1, b)
                if rem_ab <= 0:
                    continue
                min_c = int(1 / rem_ab) + 1
                # c < best_max
                for c in range(min_c, min(best_max, 200)):
                    rem_abc = rem_ab - Fraction(1, c)
                    if rem_abc <= 0:
                        continue
                    d_frac = 1 / rem_abc
                    if d_frac.denominator != 1:
                        continue
                    d = d_frac.numerator
                    if d <= c:
                        continue
                    if d >= best_max:
                        continue
                    sol = [a, b, c, d]
                    if check_solution(sol):
                        return sol
                        best_sol = sol
                        best_max = d

    return best_sol
