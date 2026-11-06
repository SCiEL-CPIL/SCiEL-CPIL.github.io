import time
from fractions import Fraction
import math
import heapq

def construct():
    target = Fraction(5, 31)
    start_time = time.time()
    time_limit = 55.0
    
    def time_ok():
        return time.time() - start_time < time_limit - 5

    # Known solution from seed
    best_solution = [8, 28, 1736]
    best_max_denom = 1736

    # Precompute harmonic numbers approximation
    gamma = 0.5772156649015329
    
    def harmonic_approx(n):
        if n <= 0:
            return 0
        return math.log(n) + gamma + 1.0/(2.0*n)
        
    def can_complete(remainder, min_denom, max_denom):
        """
        Check if it's possible to sum to 'remainder' using distinct denominators 
        in (min_denom, max_denom].
        Uses harmonic sum upper bound.
        """
        if remainder <= 0:
            return remainder == 0
        
        # Max possible sum is sum(1/d for d in range(min_denom+1, max_denom+1))
        # Approx H_max - H_min
        if max_denom <= min_denom:
            return False
            
        max_sum = harmonic_approx(max_denom) - harmonic_approx(min_denom)
        
        # If even the max possible sum is less than remainder, impossible
        if max_sum < float(remainder):
            return False
            
        return True

    def update_solution(sol):
        nonlocal best_solution, best_max_denom
        if sol is None:
            return
        m = max(sol)
        if m < best_max_denom:
            best_max_denom = m
            best_solution = list(sol)

    # Phase 1: Find all 3-term solutions 1/a + 1/b + 1/c = 5/31
    # a must be >= 7 (since 1/6 > 5/31)
    # For efficiency, only scan a up to a reasonable bound
    
    # Solve 1/b + 1/c = rem (where rem = p/q in lowest terms)
    # Returns list of (b, c) with b < c, b,c positive integers
    def solve_two_terms(rem):
        if rem <= 0:
            return []
        p = rem.numerator
        q = rem.denominator
        q2 = q * q
        sols = []
        # Factor pairs of q^2
        sqrt_q2 = int(math.isqrt(q2))
        for x in range(1, sqrt_q2 + 1):
            if q2 % x != 0:
                continue
            y = q2 // x
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
            # Verify
            if Fraction(1, b) + Fraction(1, c) == rem:
                sols.append((b, c))
        return sols
    
    # Scan a from 7 upward
    # For each a, rem = (5a-31)/(31a)
    # p = 5a-31, q = 31a
    # Factor pairs of q^2 = (31a)^2
    # This can be expensive for large a. Limit to a where solutions are likely.
    # Max denominator in solution for 3 terms: if a is small, b and c can be large.
    # We want to minimize max(b,c) so we want small a values where possible.
    
    # Let's do a complete scan but with a time budget
    max_a_scan = 200000
    for a in range(7, min(max_a_scan, 100000) + 1):
        if not time_ok():
            break
        rem = target - Fraction(1, a)
        if rem <= 0:
            break
        sols = solve_two_terms(rem)
        for (b, c) in sols:
            if b > a and c > b:
                update_solution([a, b, c])
    
    # Phase 2: Try 4-term solutions to beat the best 3-term found
    # Use Dijkstra-like approach: state is (remainder, min_denom, set_of_denoms)
    # Priority is max denominator in the path
    # Actually, let's use a targeted search
    # For 4 terms: 1/a + 1/b + 1/c + 1/d = 5/31
    # a >= 7, b > a, c > b, d > c
    
    # First, find the best 3-term solution's max denom
    # Then try to beat it with 4 terms
    
    if best_max_denom < float('inf'):
        # Try to find 4-term solutions with max < best_max_denom
        max_d_allowed = best_max_denom - 1
        
        # Dijkstra-style: states are (remainder, min_next_denom, path)
        # We want to reach remainder=0 with 4 terms, minimizing max denom
        # Use a priority queue keyed by max denominator in path
        
        # For efficiency, enumerate a and b, then solve 1/c + 1/d = rem
        # rem = 5/31 - 1/a - 1/b
        
        a_max = min(30, max_d_allowed)  # a must be small for 4 terms to help
        for a in range(7, a_max + 1):
            if not time_ok():
                break
            rem1 = target - Fraction(1, a)
            if rem1 <= 0:
                break
            # b > a, b < max_d_allowed
            # rem2 = rem1 - 1/b must be positive
            # 1/b < rem1 => b > 1/rem1
            b_min = a + 1
            b_max = min(max_d_allowed - 2, int(1 / float(rem1)) if rem1 > 0 else 0)
            
            # Let's just iterate b
            for b in range(b_min, min(b_max, max_d_allowed) + 1):
                if not time_ok():
                    break
                rem2 = rem1 - Fraction(1, b)
                if rem2 <= 0:
                    # b is too small, 1/b >= rem1
                    # Increase b
                    continue
                if rem2 > 1:
                    continue
                # Solve 1/c + 1/d = rem2 with c > b, d > c
                # rem2 = p/q
                p = rem2.numerator
                q = rem2.denominator
                q2 = q * q
                sqrt_q2 = int(math.isqrt(q2))
                found = False
                for x in range(1, sqrt_q2 + 1):
                    if not time_ok():
                        break
                    if q2 % x != 0:
                        continue
                    y = q2 // x
                    c_num = x + q
                    d_num = y + q
                    if c_num % p != 0 or d_num % p != 0:
                        continue
                    c = c_num // p
                    d = d_num // p
                    if c <= 0 or d <= 0:
                        continue
                    if c == d:
                        continue
                    if c > d:
                        c, d = d, c
                    if c > b and d > c and d <= max_d_allowed:
                        sol = [a, b, c, d]
                        update_solution(sol)
                        found = True
                        break
                if found:
                    # Can break early if we found a good solution
                    if max_d_allowed <= best_max_denom:
                        break
    
    # Phase 3: Binary search for the true minimum
    # We know there's a solution with max = best_max_denom
    # Try to find solutions with smaller max
    
    if best_solution is not None:
        low = 4
        high = best_max_denom - 1
        
        while low <= high and time_ok():
            mid = (low + high) // 2
            
            # Check if there's a solution with all denominators <= mid
            # Try 3-term first (faster)
            found = None
            
            # 3-term: 1/a + 1/b + 1/c = 5/31, a < b < c <= mid
            # a >= 7
            for a in range(7, min(mid - 2, 5000) + 1):
                if not time_ok():
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
                for a in range(7, min(25, mid - 3) + 1):
                    if not time_ok():
                        break
                    rem1 = target - Fraction(1, a)
                    if rem1 <= 0:
                        break
                    for b in range(a + 1, min(a + 30, mid - 2) + 1):
                        if not time_ok():
                            break
                        rem2 = rem1 - Fraction(1, b)
                        if rem2 <= 0:
                            continue
                        p = rem2.numerator
                        q = rem2.denominator
                        q2 = q * q
                        sqrt_q2 = int(math.isqrt(q2))
                        for x in range(1, sqrt_q2 + 1):
                            if not time_ok():
                                break
                            if q2 % x != 0:
                                continue
                            y = q2 // x
                            c_num = x + q
                            d_num = y + q
                            if c_num % p != 0 or d_num % p != 0:
                                continue
                            c = c_num // p
                            d = d_num // p
                            if c <= 0 or d <= 0:
                                continue
                            if c == d:
                                continue
                            if c > d:
                                c, d = d, c
                            if c > b and d > c and d <= mid:
                                found = [a, b, c, d]
                                break
                        if found:
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
