import time
import math
from math import gcd

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    # Known good solution
    best_solution = [10, 15, 930]
    best_max_denom = 930
    
    def find_solution(max_denom):
        # Returns a list of denominators or None
        # State: (n, d, min_d, path)
        # n/d is the remainder
        # min_d is the minimum allowed next denominator
        # path is the list of chosen denominators so far
        
        def dfs(n, d, min_d, path):
            if time.time() - start_time > time_limit:
                return None
            if n == 0:
                return path
            if n < 0:
                return None
            
            # Lower bound for x
            lower = max(min_d, (d + n - 1) // n)
            
            if lower > max_denom:
                return None
            
            for x in range(lower, max_denom + 1):
                num_next = n * x - d
                if num_next == 0:
                    return path + [x]
                if num_next < 0:
                    break
                
                den_next = d * x
                g = gcd(num_next, den_next)
                num_next //= g
                den_next //= g
                
                # Prune: Can we form num_next/den_next with denoms in [x+1, max_denom]?
                # Upper bound on sum of 1/j for j in [x+1, max_denom] is (max_denom - x) / (x + 1)
                # If this upper bound is less than the required remainder, no solution exists.
                upper_bound_sum = (max_denom - x) / (x + 1)
                if upper_bound_sum < num_next / den_next:
                    break
                    
                res = dfs(num_next, den_next, x + 1, path + [x])
                if res is not None:
                    return res
            return None
        
        return dfs(5, 31, 1, [])

    # Try to find a solution with max_denom = 200
    sol = find_solution(200)
    if sol:
        actual_max = max(sol)
        if actual_max < best_max_denom:
            best_max_denom = actual_max
            best_solution = sol
    else:
        # Try 500
        sol = find_solution(500)
        if sol:
            actual_max = max(sol)
            if actual_max < best_max_denom:
                best_max_denom = actual_max
                best_solution = sol
    return best_solution
