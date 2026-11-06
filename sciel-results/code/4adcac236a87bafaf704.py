# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def _find_3_term_solutions(target, max_a=100, max_denom=10000000, deadline=None):
    """
    Find solutions of form 1/a + 1/b + 1/c = target with a < b < c.
    Returns a list of tuples (a, b, c) or empty list.
    """
    solutions = []
    for a in range(7, max_a + 1):
        if deadline and time.time() > deadline:
            break
        ra = target - Fraction(1, a)
        if ra <= 0:
            continue
        
        # ra = num/den
        # 1/b + 1/c = num/den
        # c(num*b - den) + b*den = 0 ... no
        # 1/b + 1/c = num/den => c + b = num*bc/den => den(b+c) = num*bc
        # num*bc - den*b - den*c = 0
        # (num*b - den)(num*c - den) = den^2
        # Let X = num*b - den, Y = num*c - den. Then X*Y = den^2.
        # b = (X + den) / num, c = (Y + den) / num.
        # We need b, c to be integers, so (X+den) % num == 0 and (Y+den) % num == 0.
        # Also b > a and c > b.
        
        num = ra.numerator
        den = ra.denominator
        
        # Iterate divisors X of den^2
        # To avoid duplicates and ensure b < c, we can assume X <= Y, so X <= den.
        # Actually X*Y = den^2. If X < den, Y > den.
        # b = (X+den)/num. Since X < den, X+den < 2den.
        # We need b > a.
        
        d_sq = den * den
        # Iterate divisors X of d_sq up to sqrt(d_sq) = den
        # Optimization: only check divisors X such that (X+den) is divisible by num.
        
        # Find divisors of d_sq
        # Since den can be large (up to 31a), d_sq can be up to ~30000^2 = 900M.
        # Iterating to sqrt is ~30k. This is fast.
        
        # However, we can just iterate b directly if the range is small?
        # No, divisor method is better if den is large.
        # But for a=7, ra = 5/31 - 1/7 = (35-31)/217 = 4/217.
        # num=4, den=217. d_sq = 217^2 ~ 47000. sqrt ~ 217.
        # For a=8, ra = 5/31 - 1/8 = (40-31)/248 = 9/248.
        # num=9, den=248. d_sq ~ 60000.
        
        # Let's just iterate X from 1 to den.
        # Wait, X must be a divisor of den^2.
        # We can iterate potential divisors.
        
        # Simpler: Iterate b from a+1 to some limit?
        # If we use the divisor method:
        # X is a divisor of den^2.
        # We need (X + den) % num == 0.
        
        # Let's find all divisors of den^2.
        # For small a, den is small.
        
        # Efficient divisor check
        # We can just loop X from 1 to den. If d_sq % X == 0:
        
        for X in range(1, den + 1):
            if d_sq % X != 0:
                continue
            Y = d_sq // X
            if (X + den) % num != 0:
                continue
            if (Y + den) % num != 0:
                continue
            
            b = (X + den) // num
            c = (Y + den) // num
            
            if b <= a:
                continue
            if c <= b:
                continue
            if c > max_denom:
                continue
            
            solutions.append((a, b, c))
            
    return solutions

def _find_4_term_solutions(target, max_a=20, max_b=200, max_denom=10000000, deadline=None):
    """
    Find solutions of form 1/a + 1/b + 1/c + 1/d = target with a < b < c < d.
    """
    solutions = []
    for a in range(7, max_a + 1):
        if deadline and time.time() > deadline:
            return solutions
        ra = target - Fraction(1, a)
        if ra <= 0:
            continue
        
        # Search for b > a
        # 1/b + 1/c + 1/d = ra
        # b must be > 1/ra
        b_min = (ra.denominator + ra.numerator - 1) // ra.numerator
        if b_min <= a:
            b_min = a + 1
            
        # Limit b to keep search fast.
        # If b is too large, c and d will be large.
        # We want to minimize max(c,d).
        # Let's try b up to max_b.
        
        for b in range(b_min, max_b + 1):
            if deadline and time.time() > deadline:
                return solutions
                
            rb = ra - Fraction(1, b)
            if rb <= 0:
                break # Since b increases, 1/b decreases, rb increases? No.
                # ra is fixed. 1/b decreases as b increases.
                # So rb = ra - 1/b increases.
                # If rb <= 0, it means 1/b >= ra.
                # As b increases, 1/b decreases, so rb will eventually become positive.
                # Wait.
                # If 1/b > ra, then rb < 0. We cannot represent a negative remainder with positive reciprocals.
                # So we need 1/b < ra => b > 1/ra.
                # My b_min calculation is ceil(1/ra). So for b >= b_min, 1/b <= ra.
                # So rb >= 0.
                # If rb == 0, we have a 2-term solution 1/a+1/b=target. But we need 4 terms distinct.
                # If rb > 0, we need 1/c + 1/d = rb.
                
            num = rb.numerator
            den = rb.denominator
            
            if num == 0:
                continue # rb is 0, means 1/a+1/b=target. This is a 2-term solution, not 4.
                # But we can split 1/b into 1/b1 + 1/b2? No, we already fixed b.
                # We need 1/c + 1/d = 0, impossible.
                # So rb must be > 0.
                
            # Solve 1/c + 1/d = num/den with c > b, d > c.
            # (num*c - den)(num*d - den) = den^2
            # X = num*c - den, Y = num*d - den. X*Y = den^2.
            # c = (X+den)/num, d = (Y+den)/num.
            
            d_sq = den * den
            # Iterate X divisors of d_sq.
            # To ensure c > b:
            # (X+den)/num > b => X + den > num*b => X > num*b - den.
            
            X_min = num * b - den + 1
            if X_min < 1:
                X_min = 1
                
            # We need X <= Y => X <= den.
            # So X is in [X_min, den].
            
            # Iterate X in this range.
            # Range size is at most den.
            # den can be large?
            # ra ~ 5/31 - 1/7 ~ 0.018. rb ~ 0.018 - 1/10 ~ 0.008.
            # den ~ 1/0.008 = 125.
            # So den is small for small a,b.
            
            for X in range(X_min, den + 1):
                if d_sq % X != 0:
                    continue
                Y = d_sq // X
                if (X + den) % num != 0:
                    continue
                if (Y + den) % num != 0:
                    continue
                
                c = (X + den) // num
                d = (Y + den) // num
                
                if c <= b:
                    continue
                if d <= c:
                    continue
                if d > max_denom:
                    continue
                
                solutions.append((a, b, c, d))
                
    return solutions

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    target = Fraction(5, 31)
    best_solution = None
    best_max_denom = float('inf')
    
    # Phase 1: Find 3-term solutions
    # This is very fast.
    sols_3 = _find_3_term_solutions(target, max_a=100, max_denom=10000000, deadline=deadline)
    for s in sols_3:
        max_d = max(s)
        if max_d < best_max_denom:
            best_max_denom = max_d
            best_solution = list(s)
            
    # Phase 2: Find 4-term solutions
    # This might find a solution with a smaller max denominator than the best 3-term one.
    # Example: 1/7 + 1/8 + 1/9 + 1/... might have smaller max than 1/7+1/14+1/...
    # Actually, adding terms usually allows smaller denominators?
    # 1/7 + 1/14 + 1/42 = 6/42 + 3/42 + 1/42 = 10/42 = 5/21 != 5/31.
    # Let's rely on the search.
    
    sols_4 = _find_4_term_solutions(target, max_a=20, max_b=200, max_denom=10000000, deadline=deadline)
    for s in sols_4:
        max_d = max(s)
        if max_d < best_max_denom:
            best_max_denom = max_d
            best_solution = list(s)
            
    # Phase 3: Fallback if no solution found or if we have time
    if best_solution is None or time.time() < deadline - 5.0:
        # Try greedy with small start
        # Greedy: d = ceil(1/r)
        # This usually gives large denominators, but ensures a valid solution.
        
        # Try to improve best_solution if we have one?
        # The divisor search is exhaustive for the given bounds of a and b.
        # If we didn't find a better one, the current best is likely optimal for small a,b.
        
        if best_solution is None:
            # Fallback greedy
            r = target
            values = []
            while r > 0 and time.time() < deadline:
                d = (r.denominator + r.numerator - 1) // r.numerator
                if d in values:
                    d += 1
                    while d in values:
                        d += 1
                values.append(d)
                r -= Fraction(1, d)
                if r == 0:
                    break
            if r == 0:
                best_solution = values
                best_max_denom = max(values)
        else:
            # We have a solution. Can we do better?
            # The search spaces were max_a=100 for 3-term and max_a=20, max_b=200 for 4-term.
            # If the optimal solution has a > 100, we missed it.
            # But for 5/31, a must be >= 7.
            # 1/7 = 0.142, 5/31 = 0.161.
            # If a=7, remainder is small.
            # If a is large, say 100, 1/100 = 0.01. Remainder 0.151.
            # Then b must be ~ 7. But b > a. So b > 100.
            # 1/101 ~ 0.0099. Remainder 0.141.
            # c ~ 7. But c > b > 100.
            # So large a leads to large b,c.
            # Thus, optimal solutions should have small a.
            
            # Let's do a quick check for 5-term solutions with very small a,b,c?
            # Probably overkill. 4-term search with max_b=200 is decent.
            
            pass

    if best_solution is None:
        # Absolute fallback
        return [8, 28, 1736]
        
    # Verify
    s = sum(Fraction(1, d) for d in best_solution)
    if s != target:
        # Should not happen
        return [8, 28, 1736]
        
    return best_solution
# EVOLVE-BLOCK-END
