import time
from math import gcd
from functools import lru_cache
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    target = 479001600  # 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    target_exps = (10, 5, 2, 1, 1)
    
    # Generate all divisors of 12! that are > 12
    divisors = []
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        d = (2 ** e2) * (3 ** e3) * (5 ** e5) * (7 ** e7) * (11 ** e11)
                        if d > 12:
                            divisors.append(d)
    
    # Remove duplicates and sort
    divisors = sorted(set(divisors))
    n_divs = len(divisors)
    
    # Map value to exponents
    val_to_exps = {}
    for d in divisors:
        temp = d
        e2 = 0
        while temp % 2 == 0:
            e2 += 1
            temp //= 2
        e3 = 0
        while temp % 3 == 0:
            e3 += 1
            temp //= 3
        e5 = 0
        while temp % 5 == 0:
            e5 += 1
            temp //= 5
        e7 = 0
        while temp % 7 == 0:
            e7 += 1
            temp //= 7
        e11 = 0
        while temp % 11 == 0:
            e11 += 1
            temp //= 11
        val_to_exps[d] = (e2, e3, e5, e7, e11)
    
    # Precompute divisors with exponents, sorted in DESCENDING order
    div_exps_desc = [(val_to_exps[d], d) for d in reversed(divisors)]
    
    # To quickly find the range of divisors <= max_bound in div_exps_desc (which is descending)
    # We can store the values separately for bisect
    div_values_desc = [d for _, d in div_exps_desc]
    
    best_solution = [target]
    best_max = target

    def find_solution(rem_exps, max_bound, used_set, current_factors):
        """Find a solution with all factors <= max_bound."""
        if time.time() > deadline:
            return None
            
        if rem_exps == (0, 0, 0, 0, 0):
            return list(current_factors)
        
        if len(current_factors) >= 12:
            return None
        
        rem_prod = (2 ** rem_exps[0]) * (3 ** rem_exps[1]) * (5 ** rem_exps[2]) * (7 ** rem_exps[3]) * (11 ** rem_exps[4])
        
        if rem_prod == 1:
            return list(current_factors)
        if rem_prod < 13:
            return None
            
        # Find the index in div_values_desc where values <= max_bound start
        # div_values_desc is descending: [large, ..., small]
        # We want the first index i such that div_values_desc[i] <= max_bound
        # bisect works on ascending lists. 
        # Let's use a helper or just iterate if the list is small enough? 
        # N ~ 1000. Iterating is fast. But let's be precise.
        # We can use bisect on a reversed ascending list? 
        # divisors is ascending. 
        # Let's just iterate from the start of div_exps_desc until d <= max_bound? 
        # No, div_exps_desc is descending. So large elements are first.
        # We need to skip all d > max_bound.
        # Since it's sorted descending, we can find the position where d <= max_bound begins.
        
        # Let's use bisect on the negative values or just a linear scan if N is small.
        # N is ~1000. Linear scan is ~1000 steps. Recursion depth is small.
        # Let's stick to linear scan for simplicity and low overhead per call.
        
        for d_exps, d in div_exps_desc:
            if d > max_bound:
                continue
            if d in used_set:
                continue
            
            # Check if d divides rem_prod
            if (d_exps[0] <= rem_exps[0] and d_exps[1] <= rem_exps[1] and 
                d_exps[2] <= rem_exps[2] and d_exps[3] <= rem_exps[3] and 
                d_exps[4] <= rem_exps[4]):
                
                new_rem = (rem_exps[0] - d_exps[0], rem_exps[1] - d_exps[1],
                           rem_exps[2] - d_exps[2], rem_exps[3] - d_exps[3],
                           rem_exps[4] - d_exps[4])
                
                used_set.add(d)
                result = find_solution(new_rem, max_bound, used_set, current_factors + [d])
                used_set.remove(d)
                
                if result is not None:
                    return result
                    
        return None

    # Binary search on the maximum factor
    test_bounds = divisors
    
    lo = 0
    hi = len(test_bounds) - 1
    best_bound = target
    
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        bound = test_bounds[mid]
        
        # Try to find a solution with max factor <= bound
        used_set = set()
        solution = find_solution(target_exps, bound, used_set, [])
        
        if solution is not None:
            best_bound = bound
            best_solution = solution
            hi = mid - 1
        else:
            lo = mid + 1
            
    return best_solution
