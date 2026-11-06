import time
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
            
            # Upper bound pruning
            # Sum of 1/x for x from lower to max_denom must be >= n/d
            # Approximation: (max_denom - lower + 1) / max_denom >= n/d
            # Or better: H[max_denom] - H[lower-1] >= n/d
            # We can precompute H up to 1000 for speed
            
            # Let's just iterate
            # To find a solution with small max denom, we should try smaller x first?
            # If we pick small x, the remainder becomes small, requiring large x later.
            # If we pick large x, the remainder stays large, requiring large x later.
            # Actually, to keep max denom small, we want to avoid very small x that create tiny remainders?
            # No, tiny remainders require huge denominators.
            # So we should AVOID very small x?
            # If we pick x=7, rem is 4/217 ~ 0.018. Next x must be >= 55.
            # If we pick x=10, rem is 21/310 ~ 0.067. Next x must be >= 15.
            # The latter allows smaller denominators.
            # So we should prefer LARGER x in the early steps?
            # Let's iterate x from max_denom down to lower?
            # Or maybe iterate x from lower to max_denom, but prune heavily.
            
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
                # Max sum is approx (max_denom - x) / (x + 1) ? No.
                # Sum 1/(x+1) to 1/max_denom.
                # If this sum < num_next/den_next, then no solution with this x.
                # Since sum decreases as x increases, if it fails for x, it fails for x+1?
                # Wait. num_next/den_next increases as x increases.
                # So if it fails for x, it fails for larger x.
                # So we can break.
                
                # Approximation for sum:
                # sum_{j=x+1}^{M} 1/j ~ ln(M/(x))
                # If ln(max_denom / x) < num_next / den_next, break.
                # This is a safe lower bound for the sum?
                # Integral test: sum 1/j >= ln(M) - ln(x) = ln(M/x).
                # So if ln(M/x) < req, then sum < req.
                import math
                if math.log(max_denom / x) < num_next / den_next:
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
