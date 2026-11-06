import time
from math import gcd
from fractions import Fraction
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 90s limit

    # Known baseline solution
    best_solution = [8, 28, 1736]
    best_max_denom = 1736

    def verify(solution):
        if not solution:
            return False
        if len(solution) < 1 or len(solution) > 256:
            return False
        if len(set(solution)) != len(solution):
            return False
        if any(d < 1 or d > 10000000 for d in solution):
            return False
        s = sum(Fraction(1, d) for d in solution)
        return s == Fraction(5, 31)

    def find_solution_mitm(limit):
        """
        Meet-in-the-middle: find distinct denominators in [1, limit] summing to 5/31.
        Strategy: enumerate all 2-term and 3-term sums, then look for complementary pairs.
        """
        if time.time() - start_time > time_limit - 2.0:
            return None
        
        target = Fraction(5, 31)
        
        # Precompute all 2-term sums: 1/a + 1/b where 1 <= a < b <= limit
        # Store: (num, den) -> list of (a, b)
        two_term_sums = defaultdict(list)
        
        # Precompute all 3-term sums: 1/a + 1/b + 1/c where 1 <= a < b < c <= limit
        # Store: (num, den) -> list of (a, b, c)
        three_term_sums = defaultdict(list)
        
        # Enumerate 2-term sums
        for a in range(1, limit + 1):
            if time.time() - start_time > time_limit - 2.0:
                return None
            for b in range(a + 1, limit + 1):
                frac = Fraction(1, a) + Fraction(1, b)
                key = (frac.numerator, frac.denominator)
                two_term_sums[key].append((a, b))
        
        # Enumerate 3-term sums (this is O(limit^3) which may be too slow for large limit)
        # Only do this if limit is small enough
        if limit <= 200:
            for a in range(1, limit + 1):
                if time.time() - start_time > time_limit - 2.0:
                    return None
                for b in range(a + 1, limit + 1):
                    for c in range(b + 1, limit + 1):
                        frac = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
                        key = (frac.numerator, frac.denominator)
                        three_term_sums[key].append((a, b, c))
        
        # Now search for combinations:
        # Case 1: 1-term + 2-term = 5/31
        # 1/d + (1/a + 1/b) = 5/31 where d < a < b
        for d in range(1, limit + 1):
            if time.time() - start_time > time_limit - 2.0:
                return None
            remainder = target - Fraction(1, d)
            if remainder <= 0:
                continue
            key = (remainder.numerator, remainder.denominator)
            if key in two_term_sums:
                for (a, b) in two_term_sums[key]:
                    if a > d and b > a:
                        return [d, a, b]
        
        # Case 2: 1-term + 3-term = 5/31 (only if we computed 3-term sums)
        if three_term_sums:
            for d in range(1, limit + 1):
                if time.time() - start_time > time_limit - 2.0:
                    return None
                remainder = target - Fraction(1, d)
                if remainder <= 0:
                    continue
                key = (remainder.numerator, remainder.denominator)
                if key in three_term_sums:
                    for (a, b, c) in three_term_sums[key]:
                        if a > d and b > a and c > b:
                            return [d, a, b, c]
        
        # Case 3: 2-term + 2-term = 5/31
        # (1/a + 1/b) + (1/c + 1/d) = 5/31 where a < b < c < d
        # This is more complex; skip for now and rely on DFS fallback
        
        return None

    def find_solution_dfs(limit):
        """
        DFS fallback for when MITM doesn't work or limit is too large for MITM.
        """
        def dfs(num, den, min_d, current_set):
            if time.time() - start_time > time_limit - 1.0:
                return None
            
            if num == 0:
                return current_set
            
            if num < 0:
                return None
            
            lower_bound = (den + num - 1) // num
            if lower_bound < min_d:
                lower_bound = min_d
            
            if lower_bound > limit:
                return None
            
            for d in range(lower_bound, limit + 1):
                if time.time() - start_time > time_limit - 1.0:
                    return None
                
                if den > num * d:
                    continue
                
                next_num = num * d - den
                next_den = den * d
                
                g = gcd(next_num, next_den)
                if g > 1:
                    next_num //= g
                    next_den //= g
                
                res = dfs(next_num, next_den, d + 1, current_set + [d])
                if res is not None:
                    return res
                    
            return None
        
        return dfs(5, 31, 1, [])

    # Binary search on the maximum denominator bound
    current_limit = best_max_denom - 1
    
    while current_limit >= 1:
        if time.time() - start_time > time_limit - 2.0:
            break
        
        # Try MITM first if limit is small enough
        sol = None
        if current_limit <= 300:
            sol = find_solution_mitm(current_limit)
        
        # Fall back to DFS if MITM didn't find a solution or limit is too large
        if sol is None:
            sol = find_solution_dfs(current_limit)
        
        if sol is not None:
            actual_max = max(sol)
            if actual_max < best_max_denom:
                best_max_denom = actual_max
                best_solution = sol
            current_limit = actual_max - 1
        else:
            break
        
        if current_limit < 7:
            break

    if verify(best_solution):
        return best_solution
    
    return [8, 28, 1736]
