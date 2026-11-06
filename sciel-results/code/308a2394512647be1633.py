import time
from fractions import Fraction
import math

def construct():
    target = Fraction(5, 31)
    start_time = time.time()
    time_limit = 55.0
    
    best_solution = None
    best_max_denom = float('inf')
    
    def update_solution(sol):
        nonlocal best_solution, best_max_denom
        if sol is None:
            return
        m = max(sol)
        if m < best_max_denom:
            best_max_denom = m
            best_solution = list(sol)
    
    def solve_two_terms(remainder):
        """Solve 1/b + 1/c = remainder with b < c, distinct positive integers."""
        if remainder <= 0:
            return []
        p = remainder.numerator
        q = remainder.denominator
        # (pb - q)(pc - q) = q^2
        q2 = q * q
        solutions = []
        x_max = int(math.isqrt(q2))
        for x in range(1, x_max + 1):
            if q2 % x != 0:
                continue
            y = q2 // x
            if x > y:
                break
            b_num = x + q
            c_num = y + q
            if b_num % p != 0 or c_num % p != 0:
                continue
            b = b_num // p
            c = c_num // p
            if b <= 0 or c <= 0:
                continue
            if b == c:
                continue
            if b > c:
                b, c = c, b
            if Fraction(1, b) + Fraction(1, c) == remainder:
                solutions.append((b, c))
        return solutions
    
    # Try 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # a >= 7 (since 1/6 > 5/31)
    # For each a, solve 1/b + 1/c = 5/31 - 1/a with a < b < c
    max_a_scan = 100000
    for a in range(7, min(max_a_scan, 50000) + 1):
        if time.time() - start_time > time_limit - 10:
            break
        rem = target - Fraction(1, a)
        if rem <= 0:
            break
        sols = solve_two_terms(rem)
        for (b, c) in sols:
            if b > a and c > b:
                update_solution([a, b, c])
    
    # Try 4-term solutions with bounded DFS
    def dfs_bounded(remainder, min_denom, current_set, max_denom, terms_left):
        if time.time() - start_time > time_limit - 5:
            return None
        if remainder == 0:
            if terms_left == 0:
                return current_set
            return None
        if remainder < 0:
            return None
        if terms_left == 0:
            return None
        
        # Lower bound
        lower = (remainder.denominator + remainder.numerator - 1) // remainder.numerator
        if lower < min_denom + 1:
            lower = min_denom + 1
        if lower > max_denom:
            return None
        
        if terms_left == 1:
            if remainder.numerator == 1:
                d = remainder.denominator
                if d > min_denom and d <= max_denom:
                    return current_set + [d]
            return None
        
        # Limit iteration range for speed
        iter_limit = min(max_denom, lower + 100)
        for d in range(lower, iter_limit + 1):
            if time.time() - start_time > time_limit - 5:
                return None
            next_rem = remainder - Fraction(1, d)
            if next_rem < 0:
                continue
            res = dfs_bounded(next_rem, d, current_set + [d], max_denom, terms_left - 1)
            if res is not None:
                return res
        return None
    
    if best_max_denom < float('inf'):
        max_d = best_max_denom - 1
        if max_d >= 4:
            for a in range(7, min(50, max_d) + 1):
                if time.time() - start_time > time_limit - 10:
                    break
                rem1 = target - Fraction(1, a)
                if rem1 <= 0:
                    break
                for b in range(a + 1, min(a + 50, max_d) + 1):
                    if time.time() - start_time > time_limit - 5:
                        break
                    rem2 = rem1 - Fraction(1, b)
                    if rem2 <= 0:
                        break
                    res = dfs_bounded(rem2, b, [a, b], max_d, 2)
                    if res is not None:
                        update_solution(res)
    
    # Binary search for minimum max denominator
    if best_solution is not None:
        low = 1
        high = best_max_denom - 1
        
        while low <= high and time.time() - start_time < time_limit - 5:
            mid = (low + high) // 2
            found = None
            
            # Try 3-term solutions with max <= mid
            for a in range(7, min(50000, mid) + 1):
                if time.time() - start_time > time_limit - 5:
                    break
                rem = target - Fraction(1, a)
                if rem <= 0:
                    break
                sols = solve_two_terms(rem)
                for (b, c) in sols:
                    if b > a and c > b and c <= mid:
                        found = [a, b, c]
                        break
                if found:
                    break
            
            # Try 4-term if not found
            if found is None and mid >= 10:
                max_d = mid
                for a in range(7, min(50, max_d) + 1):
                    if time.time() - start_time > time_limit - 3:
                        break
                    rem1 = target - Fraction(1, a)
                    if rem1 <= 0:
                        break
                    for b in range(a + 1, min(a + 30, max_d) + 1):
                        if time.time() - start_time > time_limit - 3:
                            break
                        rem2 = rem1 - Fraction(1, b)
                        if rem2 <= 0:
                            break
                        res = dfs_bounded(rem2, b, [a, b], max_d, 2)
                        if res is not None:
                            found = res
                            break
                    if found:
                        break
            
            if found is not None:
                update_solution(found)
                high = mid - 1
            else:
                low = mid + 1
    
    if best_solution is None:
        best_solution = [8, 28, 1736]
    
    return best_solution
