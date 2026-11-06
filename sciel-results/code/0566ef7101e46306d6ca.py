"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    target = Fraction(5, 31)
    
    # We want to minimize the largest denominator.
    # The current seed uses greedy which likely gives a large max denominator.
    # Let's try to find a representation with small denominators.
    
    # Strategy:
    # 1. Try to find a representation using denominators from a small set [1..N].
    # 2. Use backtracking with pruning.
    # 3. Since we want to minimize the MAX denominator, we can iterate on the max allowed denominator M.
    #    However, searching for M=1, 2, ... is expensive if we don't have a good lower bound.
    #    Instead, we can do a depth-first search that tries to keep denominators small.
    
    # A better approach for "minimize largest denominator":
    # We can use a branch and bound search.
    # State: current remainder r, current max denominator used so far (or just the list of used), 
    #        and the smallest denominator allowed next (to ensure distinctness and order).
    
    # Actually, distinctness is required.
    # Let's maintain a sorted list of denominators.
    
    # Heuristic: To minimize the max, we should prefer smaller denominators early?
    # Not necessarily. But we can bound the search.
    
    # Let's try a specific structure first.
    # 5/31 is close to 1/6 (0.1666) -> 5/31 = 0.16129
    # 1/6 = 0.16666
    # 5/31 - 1/6 = (30 - 31) / 186 = -1/186. Negative.
    # So 1/6 is too big.
    # 1/7 = 0.1428
    # 5/31 - 1/7 = (35 - 31) / 217 = 4/217.
    # 4/217 is approx 0.0184.
    # 1/54 = 0.0185.
    # 4/217 - 1/54 = (216 - 217) / (217*54) = -1/11718.
    # So 1/7 + 1/54 is slightly too big.
    # 4/217 - 1/55 = (220 - 217) / (217*55) = 3 / 11935.
    # 3/11935 is approx 0.00025.
    # 1/3978 = 0.000251.
    # 3/11935 - 1/3978 = ...
    
    # Let's run a search.
    # We will search for a set of distinct denominators d1 < d2 < ... < dk such that sum(1/di) = 5/31.
    # We want to minimize dk.
    
    # We can iterate k from 1 to 10 (since 256 is max, but small k is likely for small max).
    # Actually, if we minimize max, we might need more terms if small denominators don't fit well.
    # But usually, fewer terms with balanced sizes are better for minimizing max?
    # No, 1/2 + 1/3 + 1/6 = 1. Max 6.
    # 1/100 + ... needs many terms to reach 0.5.
    # So likely a small number of terms (3-6) with moderate denominators.
    
    # Let's use a recursive search that tries to complete the sum.
    # Pruning:
    # 1. If r <= 0, fail (if r==0 and we have terms, success).
    # 2. If we have chosen 'chosen' denominators, and the remaining sum must be achieved with 'max_terms - chosen' terms.
    # 3. The next denominator d must be > last_denominator.
    # 4. To minimize max, we can try denominators in increasing order.
    # 5. Bound: If we pick d, the max denominator will be at least d.
    #    We can keep track of the best max denominator found so far (global best).
    #    If current d >= best_max, we can prune this branch? 
    #    Wait, if we are in a branch where we already picked a denominator larger than best_max, we prune.
    #    If we are picking d, and d > best_max, we prune.
    #    This requires we find a valid solution first.
    
    # Let's find ANY valid solution first to set the initial best_max.
    # The greedy method gives one.
    
    def greedy_solution():
        r = target
        vals = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d > 10000000:
                return None
            vals.append(d)
            r -= Fraction(1, d)
        return vals
    
    best_solution = greedy_solution()
    if best_solution is None:
        return [1] # Fallback, though 1 is not distinct from others if used, but 1/1 != 5/31. 
        # Actually if greedy fails, we return something valid?
        # The task says "Return 1..256 distinct denominators... summing reciprocally to exactly 5/31".
        # If we can't find one, we must return one.
        # Greedy always works for positive rationals.
        
    current_best_max = max(best_solution)
    
    # Now search for a better solution (smaller max).
    # We will try to find a solution with max < current_best_max.
    
    # Search space:
    # Denominators must be distinct.
    # Let's try to find a solution with at most 6 terms? 
    # If we can't find one with 6 terms and max < current_best_max, maybe we need more terms?
    # But more terms usually implies larger max if we are forced to use small ones?
    # No, 1/1000 + 1/1001 ... sum is small.
    # To sum to 5/31 (~0.16), if we use many small denominators, they add up fast.
    # 1/2 = 0.5 (too big).
    # 1/3 = 0.33 (too big).
    # 1/4 = 0.25 (too big).
    # 1/5 = 0.2 (too big).
    # 1/6 = 0.166 (slightly too big).
    # So the largest term must be at least 1/7?
    # If the largest term is 1/7, the rest must sum to 5/31 - 1/7 = 4/217.
    # 4/217 is approx 0.018.
    # Next term 1/d. d must be > 7.
    # 1/8 = 0.125 (too big).
    # ...
    # 1/54 = 0.0185 (too big).
    # 1/55 = 0.0181 (ok).
    
    # So it seems we need at least 2 terms.
    # Can we do it in 2 terms?
    # 1/a + 1/b = 5/31
    # (a+b)/ab = 5/31
    # 31(a+b) = 5ab
    # 5ab - 31a - 31b = 0
    # (5a - 31)(5b - 31) = 31^2 = 961
    # Factors of 961: 1, 31, 961.
    # 5a - 31 = 1 => 5a = 32 (no)
    # 5a - 31 = 31 => 5a = 62 (no)
    # 5a - 31 = 961 => 5a = 992 (no)
    # So no 2-term solution.
    
    # 3 terms?
    # 1/a + 1/b + 1/c = 5/31
    # We can search for a, b, c.
    # a must be such that 1/a < 5/31 => a > 31/5 = 6.2 => a >= 7.
    # Also a < 5/31 * 3 = 15/31? No.
    # Since 1/a is the largest term, 3/a > 5/31 => a < 3*31/5 = 18.6 => a <= 18.
    # So a in [7, 18].
    
    # For each a, let r1 = 5/31 - 1/a.
    # We need 1/b + 1/c = r1.
    # b > a.
    # 1/b < r1 => b > 1/r1.
    # 2/b > r1 => b < 2/r1.
    # So b is in [a+1, floor(2/r1)].
    
    # This is a very small search space!
    
    def solve_3_terms():
        best = None
        for a in range(7, 19):
            r1 = target - Fraction(1, a)
            if r1 <= 0:
                continue
            # b range
            b_min = a + 1
            b_max = int(2 / r1)
            if b_max < b_min:
                continue
            for b in range(b_min, b_max + 1):
                r2 = r1 - Fraction(1, b)
                if r2 <= 0:
                    continue
                # 1/c = r2 => c = 1/r2
                if r2.numerator == 1:
                    c = r2.denominator
                    if c > b:
                        max_den = c
                        if best is None or max_den < best[0]:
                            best = (max_den, [a, b, c])
        return best
    
    sol3 = solve_3_terms()
    if sol3:
        if sol3[0] < current_best_max:
            current_best_max = sol3[0]
            best_solution = sol3[1]
    
    # Can we do 4 terms?
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # a >= 7.
    # 4/a > 5/31 => a < 4*31/5 = 24.8 => a <= 24.
    # a in [7, 24].
    
    def solve_4_terms():
        best = None
        for a in range(7, 25):
            r1 = target - Fraction(1, a)
            if r1 <= 0: continue
            # b > a
            # 3/b > r1 => b < 3/r1
            b_min = a + 1
            b_max = int(3 / r1)
            if b_max < b_min: continue
            for b in range(b_min, b_max + 1):
                r2 = r1 - Fraction(1, b)
                if r2 <= 0: continue
                # c > b
                # 2/c > r2 => c < 2/r2
                c_min = b + 1
                c_max = int(2 / r2)
                if c_max < c_min: continue
                for c in range(c_min, c_max + 1):
                    r3 = r2 - Fraction(1, c)
                    if r3 <= 0: continue
                    # 1/d = r3 => d = 1/r3
                    if r3.numerator == 1:
                        d = r3.denominator
                        if d > c:
                            max_den = d
                            if best is None or max_den < best[0]:
                                best = (max_den, [a, b, c, d])
        return best
    
    if time.time() < deadline:
        sol4 = solve_4_terms()
        if sol4:
            if sol4[0] < current_best_max:
                current_best_max = sol4[0]
                best_solution = sol4[1]
    
    # Can we do 5 terms?
    # a in [7, 31*5/5 = 31] => a <= 31.
    # 5/a > 5/31 => a < 31. So a in [7, 30].
    
    def solve_5_terms():
        best = None
        for a in range(7, 31):
            if time.time() > deadline:
                return best
            r1 = target - Fraction(1, a)
            if r1 <= 0: continue
            b_min = a + 1
            b_max = int(4 / r1)
            if b_max < b_min: continue
            for b in range(b_min, b_max + 1):
                if time.time() > deadline:
                    return best
                r2 = r1 - Fraction(1, b)
                if r2 <= 0: continue
                c_min = b + 1
                c_max = int(3 / r2)
                if c_max < c_min: continue
                for c in range(c_min, c_max + 1):
                    if time.time() > deadline:
                        return best
                    r3 = r2 - Fraction(1, c)
                    if r3 <= 0: continue
                    d_min = c + 1
                    d_max = int(2 / r3)
                    if d_max < d_min: continue
                    for d in range(d_min, d_max + 1):
                        if time.time() > deadline:
                            return best
                        r4 = r3 - Fraction(1, d)
                        if r4 <= 0: continue
                        if r4.numerator == 1:
                            e = r4.denominator
                            if e > d:
                                max_den = e
                                if best is None or max_den < best[0]:
                                    best = (max_den, [a, b, c, d, e])
        return best
    
    if time.time() < deadline:
        sol5 = solve_5_terms()
        if sol5:
            if sol5[0] < current_best_max:
                current_best_max = sol5[0]
                best_solution = sol5[1]
    
    # Can we do 6 terms?
    # a in [7, 31*6/5 = 37.2] => a <= 37.
    
    def solve_6_terms():
        best = None
        for a in range(7, 38):
            if time.time() > deadline:
                return best
            r1 = target - Fraction(1, a)
            if r1 <= 0: continue
            b_min = a + 1
            b_max = int(5 / r1)
            if b_max < b_min: continue
            for b in range(b_min, b_max + 1):
                if time.time() > deadline:
                    return best
                r2 = r1 - Fraction(1, b)
                if r2 <= 0: continue
                c_min = b + 1
                c_max = int(4 / r2)
                if c_max < c_min: continue
                for c in range(c_min, c_max + 1):
                    if time.time() > deadline:
                        return best
                    r3 = r2 - Fraction(1, c)
                    if r3 <= 0: continue
                    d_min = c + 1
                    d_max = int(3 / r3)
                    if d_max < d_min: continue
                    for d in range(d_min, d_max + 1):
                        if time.time() > deadline:
                            return best
                        r4 = r3 - Fraction(1, d)
                        if r4 <= 0: continue
                        e_min = d + 1
                        e_max = int(2 / r4)
                        if e_max < e_min: continue
                        for e in range(e_min, e_max + 1):
                            if time.time() > deadline:
                                return best
                            r5 = r4 - Fraction(1, e)
                            if r5 <= 0: continue
                            if r5.numerator == 1:
                                f = r5.denominator
                                if f > e:
                                    max_den = f
                                    if best is None or max_den < best[0]:
                                        best = (max_den, [a, b, c, d, e, f])
        return best
    
    if time.time() < deadline:
        sol6 = solve_6_terms()
        if sol6:
            if sol6[0] < current_best_max:
                current_best_max = sol6[0]
                best_solution = sol6[1]
    
    # If we found a better solution, return it.
    # Otherwise return the greedy one.
    return best_solution

# EVOLVE-BLOCK-END
