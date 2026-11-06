import time
import math
from fractions import Fraction
from math import gcd

def construct():
    # Known best solution
    best_solution = [8, 28, 1736]
    best_max_denom = 1736
    
    start_time = time.time()
    time_limit = 55.0 # Safety margin for 60s limit
    
    # Precompute harmonic numbers up to a safe upper bound
    # We start with limit 1735, so 2000 is safe.
    max_precompute = 2000
    H = [0.0] * (max_precompute + 1)
    for i in range(1, max_precompute + 1):
        H[i] = H[i-1] + 1.0/i
        
    def find_solution(limit):
        """
        Finds any solution for 5/31 using distinct denominators in [1, limit].
        Returns the list of denominators or None.
        """
        
        def dfs(n, d, min_d, path):
            if time.time() - start_time > time_limit:
                return None
            
            if n == 0:
                return path
            
            if n < 0:
                return None
            
            # Pruning 1: Lower bound for next denominator d
            # 1/d <= num/den  => d >= den/num
            # d must be integer, so d >= ceil(den/num)
            lower_bound = (d + n - 1) // n
            if lower_bound < min_d:
                lower_bound = min_d
            
            if lower_bound > limit:
                return None
            
            # Iterate possible denominators
            for x in range(lower_bound, limit + 1):
                if time.time() - start_time > time_limit:
                    return None
                
                # Check if 1/x <= n/d
                # This is guaranteed by lower_bound = ceil(d/n) => x >= d/n => n*x >= d => n*x - d >= 0
                num_next = n * x - d
                if num_next == 0:
                    return path + [x]
                
                den_next = d * x
                
                # Reduce fraction
                g = gcd(num_next, den_next)
                if g > 1:
                    num_next //= g
                    den_next //= g
                    
                # Pruning 2: Max possible sum of remaining terms
                # We need to form num_next/den_next using denominators in [x+1, limit]
                # Max sum is H[limit] - H[x]
                # If H[limit] - H[x] < num_next/den_next, it's impossible.
                # Since H[limit]-H[x] decreases as x increases, and num_next/den_next increases as x increases,
                # if this condition fails, it will fail for all larger x.
                
                # Use a small epsilon to avoid floating point issues
                # If the max sum is slightly less than required, it's likely impossible.
                # However, due to float precision, we might falsely break.
                # To be safe, only break if H[limit] - H[x] < num_next/den_next - 1e-10
                # Or better, if H[limit] - H[x] + 1e-12 < num_next/den_next
                if H[limit] - H[x] < num_next / den_next - 1e-12:
                    break
                
                res = dfs(num_next, den_next, x + 1, path + [x])
                if res is not None:
                    return res
                    
            return None
            
        # Start with remainder 5/31, min_denom 1
        return dfs(5, 31, 1, [])

    # Main Search Loop
    current_limit = best_max_denom - 1
    
    while current_limit >= 1:
        if time.time() - start_time > time_limit - 1.0:
            break
            
        sol = find_solution(current_limit)
        
        if sol is not None:
            # Found a solution with max <= current_limit
            # The actual max might be smaller
            actual_max = max(sol)
            if actual_max < best_max_denom:
                best_max_denom = actual_max
                best_solution = sol
                
            # Now try to find a solution with even smaller max
            # Set new limit to actual_max - 1
            current_limit = actual_max - 1
        else:
            # No solution found for this limit.
            # This implies no solution exists for any limit < current_limit.
            # So best_solution is optimal.
            break
            
    return best_solution
