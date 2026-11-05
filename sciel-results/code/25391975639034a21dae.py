import time
from fractions import Fraction
from math import isqrt, gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    target = Fraction(5, 31)
    
    # 1. Find best 3-term solution
    best_max = 10000000
    best_solution = None
    
    # Iterate a, b to find c
    # 1/a + 1/b + 1/c = 5/31
    # 1/c = 5/31 - 1/a - 1/b
    # c = 1 / (5/31 - 1/a - 1/b)
    
    # a must be >= 7
    # b > a
    # c > b
    
    # We want to minimize max(a,b,c).
    # So we can iterate a, b and check if c is integer and c > b.
    # Then update best_max.
    
    # Optimization: bound a and b.
    # If a is large, 1/a is small.
    # Max possible value for 1/a + 1/b is < 5/31.
    # a > 31/5 = 6.2 => a >= 7.
    # If a=7, 1/b < 5/31 - 1/7 = 4/217 => b > 217/4 = 54.25 => b >= 55.
    # If a=8, 1/b < 5/31 - 1/8 = 9/248 => b > 248/9 = 27.5 => b >= 28.
    # We found 8, 28, 1736.
    
    # Let's iterate a from 7 to 200 (arbitrary limit, since 1/a becomes small)
    # and b from a+1 to 2000.
    
    for a in range(7, 200):
        if time.time() > deadline: break
        r_a = target - Fraction(1, a)
        if r_a <= 0: continue
        
        # 1/b + 1/c = r_a
        # b > a
        # 1/b < r_a => b > 1/r_a
        b_min = max(a + 1, r_a.denominator // r_a.numerator + 1)
        
        # We also want b to be not too large.
        # If b is very large, c will be close to b.
        # Let's iterate b up to 5000.
        for b in range(b_min, 5000):
            if time.time() > deadline: break
            r_b = r_a - Fraction(1, b)
            if r_b <= 0: continue
            
            # c = 1 / r_b
            if r_b.numerator == 1:
                c = r_b.denominator
                if c > b:
                    max_den = max(a, b, c)
                    if max_den < best_max:
                        best_max = max_den
                        best_solution = [a, b, c]
            # else c is not integer
            
    # 2. Search for solutions with max denominator < best_max
    # Iterate M from 7 to best_max - 1
    # For each M, check if a solution exists with all denominators <= M
    
    def dfs_check(r, min_den, M, current, depth):
        nonlocal best_max, best_solution
        if time.time() > deadline:
            return False
        if r == 0:
            if current:
                m = max(current)
                if m <= M:
                    best_max = m
                    best_solution = list(current)
                    return True
            return False
        if r < 0:
            return False
        if depth > 10: # Limit depth for small M search
            return False
            
        # Pruning: r < 1/M
        if r < Fraction(1, M):
            return False
            
        d_start = max(min_den, (r.denominator + r.numerator - 1) // r.numerator)
        if d_start > M:
            return False
            
        # Max sum prune
        # Max sum of k terms starting from d_start
        # We have at most M - d_start + 1 terms available
        # But depth limit is 10
        max_terms = min(10 - depth, M - d_start + 1)
        if max_terms <= 0:
            return False
        max_sum = sum(Fraction(1, d_start + i) for i in range(max_terms))
        if r > max_sum:
            return False
            
        for d in range(d_start, M + 1):
            current.append(d)
            if dfs_check(r - Fraction(1, d), d + 1, M, current, depth + 1):
                current.pop()
                return True
            current.pop()
        return False

    # We want the SMALLEST M.
    # So iterate M from 7 upwards.
    # But if best_max is 1736, and the answer is 1736, we will check 7..1735.
    # This might be slow if DFS is slow.
    # However, for small M, DFS is fast.
    # For large M, if no solution exists, it might explore a lot.
    # But we have the "Max Sum" prune.
    
    # Let's try iterating M from 7 to best_max - 1.
    for M in range(7, best_max):
        if time.time() > deadline:
            break
        # Optional: skip M if it's not a "likely" candidate?
        # Just run DFS.
        if dfs_check(target, 7, M, [], 0):
            # Found a solution with max <= M
            # Since we iterate M upwards, this is the minimum M.
            # But wait, dfs_check updates best_max to the actual max of the found solution.
            # The found solution might have max < M.
            # But we are looking for the minimum possible max denominator.
            # If a solution exists with max <= M, and we are iterating M upwards,
            # the first M for which a solution exists is the minimum max denominator?
            # No. If a solution exists with max=10, it also exists with max<=11, max<=12, etc.
            # So the first M that returns True is the minimum max denominator.
            break
            
    if best_solution is not None:
        # Verify
        s = sum(Fraction(1, d) for d in best_solution)
        if s == target and len(set(best_solution)) == len(best_solution) and all(1 <= d <= 10000000 for d in best_solution):
            return best_solution
            
    # Fallback
    return [8, 28, 1736]
