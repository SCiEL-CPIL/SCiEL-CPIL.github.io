import time
from math import gcd

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit

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
        from fractions import Fraction
        s = sum(Fraction(1, d) for d in solution)
        return s == Fraction(5, 31)

    def find_3_term_solutions():
        solutions = []
        # a ranges from 7 to 18
        for a in range(7, 19):
            N = 5 * a - 31
            D = 31 * a
            if N <= 0:
                continue
            D2 = D * D
            # Find divisors X of D2 such that X <= D2//X and X % N == (-D) % N
            target_mod = (-D) % N
            # Iterate divisors up to sqrt(D2)
            # D2 is at most 31*18 squared = 558^2 = 311364
            # sqrt is ~558
            for X in range(1, int(D2**0.5) + 1):
                if D2 % X != 0:
                    continue
                Y = D2 // X
                # Check X
                if X % N == target_mod:
                    b = (X + D) // N
                    c = (Y + D) // N
                    if b > a and c > b:
                        solutions.append((a, b, c))
                # Check Y if Y != X
                if Y != X and Y % N == target_mod:
                    b = (Y + D) // N
                    c = (X + D) // N
                    if b > a and c > b:
                        solutions.append((a, b, c))
        return solutions

    sols = find_3_term_solutions()
    if sols:
        # Find the one with minimum max denominator
        best_3 = min(sols, key=lambda x: x[2])
        if best_3[2] < best_max_denom:
            best_solution = list(best_3)
            best_max_denom = best_3[2]

    # Try to find a 4-term solution with max denom < best_max_denom
    # using DFS with limit best_max_denom - 1
    def find_solution_with_limit(limit):
        def dfs(num, den, min_d, current_set):
            if time.time() - start_time > time_limit:
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
            
            # Prune: max possible sum
            # 1/lower_bound + 1/(lower_bound+1) + ... 
            # Approximation: 1/lower_bound * (limit - lower_bound + 1)
            # If this is < num/den, prune? No, harmonic series converges.
            # Simple prune: if 1/lower_bound > num/den, return None (handled by lower_bound)
            
            for d in range(lower_bound, limit + 1):
                if time.time() - start_time > time_limit:
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

    # Only run DFS if we have time and want to check for 4-term improvement
    if best_max_denom > 100: # If best is already very small, skip
        # Try limit best_max_denom - 1
        limit = best_max_denom - 1
        if limit > 0:
            sol = find_solution_with_limit(limit)
            if sol is not None:
                actual_max = max(sol)
                if actual_max < best_max_denom:
                    best_solution = sol
                    best_max_denom = actual_max

    if verify(best_solution):
        return best_solution
    
    return [8, 28, 1736]
