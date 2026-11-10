import math
import time
import bisect

def construct():
    start_time = time.time()
    deadline = 85  # safety margin before 90s kill
    
    N = math.factorial(12)  # 479001600
    
    # Global best
    best_max = N
    best_factors = [N]
    
    # Precompute all divisors of N
    # N = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    divisors = [1]
    primes_exp = [(2, 10), (3, 5), (5, 2), (7, 1), (11, 1)]
    for p, e in primes_exp:
        new_divs = []
        for d in divisors:
            for i in range(e + 1):
                new_divs.append(d * (p ** i))
        divisors = new_divs
    
    # Filter: > 12
    valid_divisors = sorted([d for d in divisors if d > 12])
    
    def dfs(remaining_target, last_factor, count, path):
        nonlocal best_max, best_factors
        
        if time.time() - start_time > deadline:
            return False
        
        if count == 0:
            if remaining_target == 1:
                max_f = path[-1] if path else 0
                if max_f < best_max:
                    best_max = max_f
                    best_factors = list(path)
            return True
        
        if count == 1:
            x = remaining_target
            if x > last_factor and x > 12:
                if x < best_max:
                    best_max = x
                    best_factors = path + [x]
            return True

        # Lower bound for next factor x:
        # x must be > last_factor
        # Also, the product of the remaining 'count' factors (including x) must be <= remaining_target
        # x * (x+1) * ... * (x+count-1) <= remaining_target
        
        lower_bound = max(last_factor + 1, 13)
        
        # Find upper bound for x
        # x * (x+1) * ... * (x+count-1) <= remaining_target
        x_upper = lower_bound
        while True:
            prod_test = 1
            for i in range(count):
                prod_test *= (x_upper + i)
                if prod_test > remaining_target:
                    break
            if prod_test > remaining_target:
                break
            x_upper += 1
            if x_upper > 10000:
                break
        x_upper = min(x_upper - 1, remaining_target)
        
        # Also bound by best_max - 1
        if best_max < N:
            x_upper = min(x_upper, best_max - 1)
        
        if x_upper < lower_bound:
            return True
        
        # Get divisors of remaining_target in range [lower_bound, x_upper]
        # We have valid_divisors sorted.
        # Find index of first divisor >= lower_bound
        start_idx = bisect.bisect_left(valid_divisors, lower_bound)
        end_idx = bisect.bisect_right(valid_divisors, x_upper)
        
        # Iterate through divisors in that range
        for i in range(start_idx, end_idx):
            if time.time() - start_time > deadline:
                return False
            
            x = valid_divisors[i]
            
            # Check if x divides remaining_target
            if remaining_target % x != 0:
                continue
            
            rem_target = remaining_target // x
            
            # Check if rem_target can be formed by (count-1) distinct integers > x
            # Necessary condition: rem_target >= (x+1)(x+2)...(x+count-1)
            min_rem = 1
            for j in range(1, count):
                min_rem *= (x + j)
                if min_rem > rem_target:
                    break
            if rem_target < min_rem:
                continue
            
            # Optional: Check if rem_target <= (best_max-1)(best_max-2)... 
            # This is an upper bound on the product if we want to beat best_max
            if best_max < N:
                max_rem = 1
                for j in range(count - 1):
                    max_rem *= (best_max - 1 - j)
                    if max_rem > rem_target:
                        break
                if rem_target > max_rem:
                    continue
            
            path.append(x)
            dfs(rem_target, x, count - 1, path)
            path.pop()
            
            if best_max <= 15:
                return True
        
        return True

    # Try k values in order of likelihood to find a good solution quickly
    # Geometric mean for k=7 is ~14.5. k=6 is ~28. k=5 is ~55.
    # We want to minimize the MAX factor. 
    # If we use more factors, the max factor tends to be smaller (closer to GM).
    # But we are constrained by distinctness and > 12.
    # Let's try k=8, 7, 6, 9, 5...
    
    for k in [8, 7, 6, 9, 5, 10, 4, 11, 3, 12, 2, 1]:
        if time.time() - start_time > deadline:
            break
        dfs(N, 12, k, [])
        if best_max <= 15:
            break

    return best_factors
