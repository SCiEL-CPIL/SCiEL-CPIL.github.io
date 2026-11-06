import time
from math import gcd

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
        # Use integer arithmetic for verification to be safe and fast
        # Sum 1/d_i = 5/31
        # num/den = 5/31
        num = 0
        den = 1
        for d in solution:
            # num/den + 1/d = (num*d + den) / (den*d)
            new_num = num * d + den
            new_den = den * d
            g = gcd(new_num, new_den)
            num = new_num // g
            den = new_den // g
        return num == 5 and den == 31

    def find_solution_with_limit(limit):
        """
        Finds any solution for 5/31 using distinct denominators in [1, limit].
        Returns the list of denominators or None.
        Uses DFS with pruning.
        """
        
        def dfs(num, den, min_d, current_set):
            # Time check
            if time.time() - start_time > time_limit:
                return None
            
            if num == 0:
                return current_set
            
            if num < 0:
                return None
            
            # Lower bound for next denominator d
            # We need 1/d <= num/den => d >= den/num
            lower_bound = (den + num - 1) // num
            if lower_bound < min_d:
                lower_bound = min_d
            
            if lower_bound > limit:
                return None
            
            # Upper bound check:
            # The maximum sum we can add with denominators >= lower_bound is roughly
            # sum_{j=lower_bound}^{limit} 1/j
            # A simple upper bound is (limit - lower_bound + 1) / lower_bound
            # If this is less than num/den, we can't reach the target.
            # However, this bound is quite loose. A tighter bound is ln(limit/lower_bound).
            # Let's use a simple check: if 1/lower_bound * (limit - lower_bound + 1) < num/den, prune.
            # Actually, sum_{j=k}^{N} 1/j <= (N-k+1)/k.
            # If (limit - lower_bound + 1) / lower_bound < num / den, then it's impossible.
            # (limit - lower_bound + 1) * den < num * lower_bound
            if (limit - lower_bound + 1) * den < num * lower_bound:
                return None

            for d in range(lower_bound, limit + 1):
                if time.time() - start_time > time_limit:
                    return None
                
                # Check if 1/d <= num/den
                # den <= num * d
                if den > num * d:
                    continue
                
                next_num = num * d - den
                if next_num == 0:
                    return current_set + [d]
                
                if next_num < 0:
                    break
                
                next_den = den * d
                
                g = gcd(next_num, next_den)
                if g > 1:
                    next_num //= g
                    next_den //= g
                
                res = dfs(next_num, next_den, d + 1, current_set + [d])
                if res is not None:
                    return res
                    
            return None
        
        # Start with remainder 5/31, min_denom 1
        return dfs(5, 31, 1, [])

    # Main Search Loop
    # We want to minimize the largest denominator.
    # Start by trying to find a solution with max denom < current best.
    current_limit = best_max_denom - 1
    
    while current_limit >= 1:
        if time.time() - start_time > time_limit - 1.0:
            break
        
        sol = find_solution_with_limit(current_limit)
        
        if sol is not None:
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
            
    # Final verification
    if verify(best_solution):
        return best_solution
    
    # Fallback to known valid solution if something went wrong
    return [8, 28, 1736]
