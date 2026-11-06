import time
import math
from math import gcd

def construct():
    # Known best solution
    best_solution = [8, 28, 1736]
    best_max_denom = 1736
    
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    def check_two_terms(num, den, min_d, limit):
        """
        Check if num/den can be written as 1/x + 1/y with distinct x, y 
        such that min_d <= x, y <= limit.
        Returns (x, y) if found, else None.
        """
        if num == 0:
            return None # Should be handled by depth 0
        
        # (nx - d)(ny - d) = d^2
        # Let A = nx - d, B = ny - d
        # A * B = d^2
        # x = (A + d) / n, y = (B + d) / n
        
        # Iterate divisors A of d^2
        # To be efficient, iterate up to sqrt(d^2) = d
        # But d can be large.
        # However, we can just iterate A from 1 to d
        # Actually, we need A > d and B > d for x, y > 0?
        # nx = A + d. Since x > 0, nx > 0.
        # If A < d, then nx < 2d? No.
        # We need x >= min_d => nx - d >= n*min_d - d => A >= n*min_d - d.
        # We need x <= limit => nx - d <= n*limit - d => A <= n*limit - d.
        
        # Also A * B = d^2 => B = d^2 / A.
        # So A must divide d^2.
        
        # Range for A:
        # A_min = max(1, n*min_d - d)
        # A_max = min(d, n*limit - d) # Since A <= B or B <= A, we can assume A <= d
        # Actually, if A > d, then B < d. We can just iterate A <= d and check both (A, B) and (B, A) if distinct.
        
        A_min = max(1, n*min_d - d) if (n*min_d - d) > 0 else 1
        # We also need B to satisfy the bounds.
        # B = d^2 / A.
        # n*min_d - d <= B <= n*limit - d
        
        # Iterate A from A_min to d
        # Optimization: Only check divisors.
        # Finding divisors of d^2 is fast if we factor d.
        
        # For simplicity and speed, if d is small (< 10000), iterate.
        # If d is large, we might miss solutions or be slow.
        # But d is the denominator of the remainder.
        
        # Let's try iterating A from A_min to min(d, n*limit - d)
        # If d is very large, this is slow.
        # But in Egyptian fraction search, denominators grow.
        
        # Alternative: Iterate x from min_d to limit?
        # 1/y = num/den - 1/x = (num*x - den) / (den*x)
        # y = den*x / (num*x - den)
        # We need num*x - den > 0 => x > den/num.
        # And num*x - den must divide den*x.
        
        # This is O(limit - den/num).
        # If limit is 10000, this is fast.
        # If limit is 1000000, this is slow.
        
        # Let's use the divisor method if d^2 is not too big, or the linear scan if range is small.
        
        # Given we are minimizing max denom, limit is likely < 10000 in the successful branch.
        # So linear scan of x is likely faster than factoring d^2.
        
        x_start = max(min_d, den // num + 1)
        if x_start > limit:
            return None
            
        # We also need y >= min_d.
        # y = den*x / (num*x - den)
        # y >= min_d => den*x >= min_d * (num*x - den)
        # den*x >= min_d*num*x - min_d*den
        # den*min_d >= x * (min_d*num - den)
        # If min_d*num - den > 0:
        # x <= den*min_d / (min_d*num - den)
        
        x_end = limit
        denom_factor = min_d * num - den
        if denom_factor > 0:
             x_end = min(limit, (den * min_d) // denom_factor)
             
        # Also need y <= limit
        # den*x / (num*x - den) <= limit
        # den*x <= limit * (num*x - den)
        # den*x <= limit*num*x - limit*den
        # limit*den <= x * (limit*num - den)
        # If limit*num - den > 0:
        # x >= limit*den / (limit*num - den)
        
        x_start2 = (limit * den + limit * num - den - 1) // (limit * num - den) if (limit * num - den) > 0 else x_start
        x_start = max(x_start, x_start2)
        
        if x_start > x_end:
            return None
            
        for x in range(x_start, x_end + 1):
            # Check if num*x - den divides den*x
            num_x_den = num * x - den
            if num_x_den <= 0:
                continue
            if (den * x) % num_x_den == 0:
                y = (den * x) // num_x_den
                if y >= min_d and y <= limit and y != x:
                    return (x, y)
                    
        return None

    def find_solution_with_limit(limit, max_depth=7):
        """
        Finds any solution for 5/31 using distinct denominators in [1, limit].
        Returns the list of denominators or None.
        """
        
        def dfs(num, den, min_d, depth, current_set):
            # Time check
            if time.time() - start_time > time_limit:
                return None
            
            if num == 0:
                if depth == 0:
                    return current_set
                else:
                    return None # Used too few terms? No, we can stop.
            
            if num < 0:
                return None
            
            if depth == 0:
                return None
            
            # Pruning 1: Lower bound for next denominator d
            # 1/d <= num/den  => d >= den/num
            lower_bound = (den + num - 1) // num
            if lower_bound < min_d:
                lower_bound = min_d
            
            if lower_bound > limit:
                return None
            
            # Pruning 2: Check if we have enough terms
            # If depth == 1:
            #   We need 1/d = num/den => d = den/num.
            #   Check if d is integer, >= min_d, <= limit.
            if depth == 1:
                if den % num == 0:
                    d = den // num
                    if d >= min_d and d <= limit:
                        return current_set + [d]
                return None
                
            # If depth == 2:
            #   Use check_two_terms
            if depth == 2:
                res = check_two_terms(num, den, min_d, limit)
                if res:
                    return current_set + [res[0], res[1]]
                return None

            # Pruning 3: Max sum check (Float)
            # Sum of 1/k for k in [lower_bound, limit]
            # Approx: log(limit) - log(lower_bound-1)
            if lower_bound > 1:
                max_sum = math.log(limit) - math.log(lower_bound - 1)
            else:
                max_sum = math.log(limit) + 0.577
            
            if max_sum < num / den - 1e-15:
                return None
                
            # Pruning 4: Min sum check?
            # Min sum is 1/limit * depth? No, distinct.
            # Smallest sum using 'depth' distinct denominators from [limit-depth+1, limit]
            # This is a lower bound on the sum we CAN achieve? No, we need to achieve EXACTLY num/den.
            # We need num/den to be <= MaxSum.
            # We also need num/den to be >= MinSum?
            # MinSum using 'depth' terms from [lower_bound, ...]
            # To minimize the sum, we pick the largest denominators: limit, limit-1, ...
            # min_sum = sum(1/k for k in [limit-depth+1, limit])
            # If num/den < min_sum, we can't form it?
            # No, we can pick smaller denominators to increase the sum.
            # So min_sum is not a lower bound for feasibility, it's a lower bound for the sum if we are forced to use large denominators.
            # But we are not forced. We can use smaller denominators (closer to lower_bound) to increase the sum.
            # So the only bound is MaxSum.
            
            # Iterate possible denominators
            # We can restrict the range of d.
            # d must be such that the remaining sum (num/den - 1/d) can be formed by depth-1 terms from d+1 to limit.
            # Remaining sum R = (num*d - den) / (den*d).
            # Max sum for remaining terms: log(limit) - log(d)
            # So we need R <= log(limit) - log(d)
            
            # Also R > 0 => num*d > den => d > den/num.
            
            # Let's iterate d from lower_bound to limit.
            # To speed up, we can jump?
            # No, simple iteration is okay if limit is small.
            
            for d in range(lower_bound, limit + 1):
                if time.time() - start_time > time_limit:
                    return None
                
                # Check 1/d <= num/den
                if num * d < den:
                    continue
                
                next_num = num * d - den
                next_den = den * d
                
                # Reduce fraction
                g = gcd(next_num, next_den)
                if g > 1:
                    next_num //= g
                    next_den //= g
                
                # Quick prune: check if remainder can be achieved
                if next_num > 0:
                    # Max sum from d+1 to limit
                    if d + 1 > 1:
                        max_rem = math.log(limit) - math.log(d)
                    else:
                        max_rem = math.log(limit) + 0.577
                        
                    if max_rem < next_num / next_den - 1e-15:
                        continue
                        
                    # Min terms check
                    # We have depth-1 terms left.
                    # Each term is at most 1/(d+1).
                    # Max possible sum with depth-1 terms is (depth-1)/(d+1).
                    # If next_num/next_den > (depth-1)/(d+1), impossible.
                    if next_num * (d + 1) > next_den * (depth - 1):
                        continue
                        
                # Recurse
                res = dfs(next_num, next_den, d + 1, depth - 1, current_set + [d])
                if res is not None:
                    return res
                    
            return None
            
        # Start with remainder 5/31, min_denom 1, depth max_depth
        return dfs(5, 31, 1, max_depth, [])

    # Main Search Loop
    # We know 1736 works. Try smaller.
    current_limit = best_max_denom - 1
    
    while current_limit >= 1:
        if time.time() - start_time > time_limit - 1.0:
            break
            
        sol = find_solution_with_limit(current_limit)
        
        if sol is not None:
            # Found a solution with max <= current_limit
            actual_max = max(sol)
            if actual_max < best_max_denom:
                best_max_denom = actual_max
                best_solution = sol
                
            # Now try to find a solution with even smaller max
            current_limit = actual_max - 1
        else:
            # No solution found for this limit.
            # It's possible we missed it due to depth limit or pruning.
            # But if we assume the search is complete for this limit, we break.
            break
            
    return best_solution
