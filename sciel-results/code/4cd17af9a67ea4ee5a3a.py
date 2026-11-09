import math
from fractions import Fraction
import time

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 80.0


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    def check_time():
        return time.time() > deadline

    # Known solution: [8, 31, 248], max = 248
    best_solution = [8, 31, 248]
    best_max = 248

    # Verify
    s = Fraction(1, 8) + Fraction(1, 31) + Fraction(1, 248)
    assert s == TARGET, f"Known solution invalid: {s}"

    # Strategy: Try to find a 3-term solution with max < 248.
    # For 1/a + 1/b + 1/c = 5/31 with a < b < c:
    # Fix c (the largest), then 1/a + 1/b = 5/31 - 1/c = r_c.
    # r_c = p_c/q_c. Then (p_c*a - q_c)(p_c*b - q_c) = q_c^2.
    # Let u = p_c*a - q_c, v = p_c*b - q_c. Then u*v = q_c^2.
    # a = (u + q_c)/p_c, b = (v + q_c)/p_c.
    # Constraints: a >= 7, a < b < c, a and b integers.
    # a < b => u < v => u < q_c (since u*v = q_c^2).
    # a >= 7 => (u + q_c) >= 7*p_c => u >= 7*p_c - q_c.
    # b < c => (v + q_c) < c*p_c => v < c*p_c - q_c => q_c^2/u < c*p_c - q_c => u > q_c^2/(c*p_c - q_c).
    
    found_better = False
    # Iterate c from 10 up to best_max - 1
    for c in range(10, best_max):
        if check_time():
            break

        # r_c = 5/31 - 1/c
        r_c = TARGET - Fraction(1, c)
        if r_c <= 0:
            continue

        p = r_c.numerator
        q = r_c.denominator

        q2 = q * q

        # u_min: max(1, 7*p - q, floor(q2/(c*p - q)) + 1)
        # u_max: q - 1 (to ensure u < q, hence a < b)
        
        # Check if c*p > q (needed for the bound to make sense)
        if c * p <= q:
            continue
            
        u_min = max(1, 7 * p - q)
        # u > q2 / (c*p - q)
        # u >= floor(q2 / (c*p - q)) + 1
        bound = q2 // (c * p - q)
        u_min = max(u_min, bound + 1)

        u_max = q - 1

        if u_min > u_max:
            continue

        for u in range(u_min, u_max + 1):
            if check_time():
                break
            if q2 % u != 0:
                continue
            v = q2 // u
            num_a = u + q
            if num_a % p != 0:
                continue
            a = num_a // p
            if a < 7 or a >= c:
                continue
            num_b = v + q
            if num_b % p != 0:
                continue
            b = num_b // p
            if b <= a or b >= c:
                continue
            # Verify
            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == TARGET:
                if c < best_max:
                    best_max = c
                    best_solution = [a, b, c]
                    found_better = True

    # If we found a better 3-term solution, best_max is updated.
    # Now, try to find a solution with max < best_max using 4+ terms.
    # This is harder. We can use a greedy/DFS approach but it's slow.
    # However, if best_max is small (e.g., < 200), the search space is smaller.
    
    # Let's try a specific heuristic for 4 terms:
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # Fix a, b. Then 1/c + 1/d = 5/31 - 1/a - 1/b = r.
    # Same divisor method for c, d.
    
    # We iterate a from 7 to best_max-3, b from a+1 to best_max-2.
    # For each (a,b), check if we can find c,d < best_max.
    
    # This might be too slow if best_max is still 248.
    # But if we found a 3-term solution with max, say, 150, then we search for max < 150.
    
    if found_better:
        limit = best_max
        
        # Try 4-term solutions
        # a < b < c < d < limit
        for a in range(7, limit - 2):
            if check_time():
                break
            rem_a = TARGET - Fraction(1, a)
            if rem_a <= 0:
                continue
            
            for b in range(a + 1, limit - 1):
                if check_time():
                    break
                rem_b = rem_a - Fraction(1, b)
                if rem_b <= 0:
                    break
                
                # Solve 1/c + 1/d = rem_b with b < c < d < limit
                p = rem_b.numerator
                q = rem_b.denominator
                q2 = q * q
                
                # c > b, d > c
                # u = p*c - q, v = p*d - q
                # u < q (since c < d)
                # c > b => p*c - q > p*b - q => u > p*b - q
                # d < limit => p*d - q < p*limit - q => v < p*limit - q => q2/u < p*limit - q => u > q2/(p*limit - q)
                
                if p * limit <= q:
                    continue
                
                u_min = max(p * b - q + 1, q2 // (p * limit - q) + 1)
                u_max = q - 1
                
                if u_min > u_max:
                    continue
                
                for u in range(u_min, u_max + 1):
                    if check_time():
                        break
                    if q2 % u != 0:
                        continue
                    v = q2 // u
                    num_c = u + q
                    if num_c % p != 0:
                        continue
                    c = num_c // p
                    if c <= b or c >= limit:
                        continue
                    num_d = v + q
                    if num_d % p != 0:
                        continue
                    d = num_d // p
                    if d <= c or d >= limit:
                        continue
                    
                    if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == TARGET:
                        if d < best_max:
                            best_max = d
                            best_solution = [a, b, c, d]
                            found_better = True
                            limit = best_max # Update limit to search for even smaller?
                            # Note: updating limit inside loop is tricky. 
                            # But if we find a smaller one, we can break out or continue.
                            # Let's just record it. The outer loops will continue with old limit unless we break.
                            # To be safe, let's just update best and let the time guard handle it.
                            # Or break to restart? No, just continue.

    return best_solution
