import time
from fractions import Fraction
import math

def construct():
    target = Fraction(5, 31)
    
    # Start with a known solution
    best_solution = [8, 28, 1736]
    best_max_denom = 1736
    
    # Try to improve with a quick search
    start_time = time.time()
    time_limit = 55.0
    
    gamma = 0.5772156649015329
    
    def harmonic_approx(n):
        if n <= 0:
            return 0
        return math.log(n) + gamma + 1.0/(2.0*n)
    
    def can_complete(remainder, min_denom, max_denom):
        if remainder <= 0:
            return remainder == 0
        if max_denom <= min_denom:
            return False
        max_sum = harmonic_approx(max_denom) - harmonic_approx(min_denom)
        if max_sum < float(remainder):
            return False
        return True
    
    def find_solution_with_max(k):
        """Find any solution with all denominators <= k, minimizing count."""
        def dfs_local(remainder, min_denom, current_set):
            if time.time() - start_time > time_limit:
                return None
            if remainder == 0:
                return current_set
            if remainder < 0:
                return None
            if not can_complete(remainder, min_denom, k):
                return None
            lower_bound = (1 / remainder).numerator // (1 / remainder).denominator
            if lower_bound < min_denom + 1:
                lower_bound = min_denom + 1
            if lower_bound > k:
                return None
            # Try denominators in a smarter order: prefer larger denominators first
            # to keep the remainder manageable
            for d in range(k, lower_bound - 1, -1):
                if time.time() - start_time > time_limit:
                    return None
                next_rem = remainder - Fraction(1, d)
                if next_rem < 0:
                    continue
                if next_rem == 0:
                    return current_set + [d]
                res = dfs_local(next_rem, d, current_set + [d])
                if res is not None:
                    return res
            return None
        return dfs_local(target, 0, [])
    
    # Binary search for minimum K
    low = 1
    high = best_max_denom
    
    while low <= high:
        if time.time() - start_time > time_limit - 2.0:
            break
        mid = (low + high) // 2
        sol = find_solution_with_max(mid)
        if sol is not None:
            max_in_sol = max(sol)
            if max_in_sol < best_max_denom:
                best_max_denom = max_in_sol
                best_solution = sol
            high = mid - 1
        else:
            if time.time() - start_time > time_limit - 5.0:
                break
            low = mid + 1
    
    return best_solution
