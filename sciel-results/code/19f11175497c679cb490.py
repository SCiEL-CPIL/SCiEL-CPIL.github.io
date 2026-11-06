import time
import math
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit

    # Known good solution to start with
    # 5/31 = 1/8 + 1/28 + 1/1736 ?
    # 1/8 + 1/28 = 7/56 + 2/56 = 9/56
    # 5/31 - 9/56 = (280 - 279) / 1736 = 1/1736
    # Yes, [8, 28, 1736] is a valid solution. Max denom 1736.
    
    best_solution = [8, 28, 1736]
    best_max_denom = 1736

    def check_solution(denoms):
        s = Fraction(0)
        for d in denoms:
            s += Fraction(1, d)
        return s == Fraction(5, 31) and len(set(denoms)) == len(denoms) and all(1 <= d <= 10000000 for d in denoms)

    # Verify initial solution
    if not check_solution(best_solution):
        # Fallback if logic changes
        best_solution = [31, 31, 31] # Invalid, just placeholder
        best_max_denom = 31

    def find_egyptian_fraction(num, den, min_den, max_den, max_terms, used_set, current_list, remaining_time):
        """
        Finds a representation of num/den as sum of distinct unit fractions
        with denominators in [min_den, max_den].
        """
        if time.time() - start_time > remaining_time:
            return None

        # Base case: no terms left
        if max_terms == 0:
            if num == 0:
                return current_list
            return None

        # If num/den == 0, we are done (but we need exactly max_terms? No, just sum to target)
        # Actually, if num == 0, we have sum to 0. But we need distinct positive denominators.
        # If num == 0, we can only return if we've already formed the sum.
        # But the recursion structure usually implies we pick terms until num becomes 0.
        # If num == 0, we can stop.
        if num == 0:
            return current_list

        if num < 0:
            return None

        # Lower bound for next denominator d:
        # 1/d <= num/den  => d >= den/num
        # Also d >= min_den
        lower_bound = max(min_den, (den + num - 1) // num)
        
        # Upper bound for next denominator d:
        # We need to be able to form the remainder with max_terms-1 terms.
        # The smallest possible sum of max_terms-1 terms with denominators > d is
        # roughly 1/(d+1) + ... + 1/(d+max_terms-1).
        # But a tighter bound: The largest term we can add is 1/lower_bound.
        # Actually, we iterate d from lower_bound upwards.
        
        # Pruning: Max possible sum with max_terms terms starting from lower_bound
        # Approximation: max_terms / lower_bound
        # If num/den > max_terms / lower_bound, then even picking the smallest possible denominators
        # (lower_bound, lower_bound+1, ...) won't be enough? 
        # Wait, 1/lower_bound is the LARGEST term. 
        # Sum of 1/k for k from lower_bound to lower_bound+max_terms-1 is the MAX sum.
        # If num/den > that max sum, impossible.
        # But we can pick smaller denominators? No, min_den is the constraint.
        # So if num/den > sum(1/k for k in range(lower_bound, lower_bound+max_terms)), return None.
        
        # For speed, use approximation: num/den <= max_terms / lower_bound
        # This is a necessary condition.
        if num * lower_bound > den * max_terms:
            return None

        # Pruning: Min possible sum with max_terms terms ending at max_den?
        # Not useful for lower bound of d.
        
        # Iterate possible denominators d
        # We can limit the upper bound of d for this step.
        # If we pick d, remainder is num/den - 1/d = (num*d - den) / (den*d).
        # This remainder must be positive, so num*d > den => d > den/num.
        # Also, the remainder must be formable by max_terms-1 terms with denominators > d.
        
        # Let's iterate d from lower_bound to max_den.
        # Optimization: If d is too large, the remainder becomes very small,
        # requiring many small denominators? No, small denominators give large sums.
        # If remainder is small, we need large denominators.
        # Max sum of max_terms-1 terms with denominators > d is approx (max_terms-1)/(d+1).
        # If remainder > (max_terms-1)/(d+1), then we can't form it with terms > d?
        # No, we CAN form it if we pick terms close to d.
        # The MAX sum of max_terms-1 terms with denominators > d is achieved by picking d+1, d+2, ...
        # So if remainder > sum(1/k for k in range(d+1, d+max_terms)), it's impossible.
        
        # Upper bound for d in this loop:
        # We need remainder <= max_sum_remaining.
        # (num*d - den) / (den*d) <= (max_terms-1) / (d+1)
        # (num*d - den)(d+1) <= den*d*(max_terms-1)
        # num*d^2 + num*d - den*d - den <= den*d*max_terms - den*d
        # num*d^2 + d(num - den + den) - den <= den*d*max_terms
        # num*d^2 + num*d - den <= den*max_terms*d
        # num*d^2 + d(num - den*max_terms) - den <= 0
        
        # This quadratic in d gives an upper bound.
        # a = num, b = num - den*max_terms, c = -den
        # d <= (-b + sqrt(b^2 - 4ac)) / 2a
        
        a = num
        b = num - den * (max_terms - 1) # Wait, the derivation was for max_terms-1 terms remaining.
        # Let's re-derive carefully.
        # Remainder R = num/den - 1/d
        # We need R <= MaxSum(d+1, max_terms-1)
        # MaxSum approx (max_terms-1)/(d+1)
        # (num*d - den)/(den*d) <= (max_terms-1)/(d+1)
        # (num*d - den)(d+1) <= den*d*(max_terms-1)
        # num*d^2 + num*d - den*d - den <= den*(max_terms-1)*d
        # num*d^2 + d(num - den - den*(max_terms-1)) - den <= 0
        # num*d^2 + d(num - den*max_terms) - den <= 0
        
        b = num - den * max_terms
        c = -den
        
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            upper_bound_d = max_den
        else:
            sqrt_disc = math.sqrt(discriminant)
            # Roots are (-b +/- sqrt_disc) / 2a
            # Since a>0, parabola opens up. We want region <= 0, which is between roots.
            # Lower root is negative (since c<0, product of roots c/a < 0).
            # Upper root is (-b + sqrt_disc) / 2a
            upper_root = (-b + sqrt_disc) / (2*a)
            upper_bound_d = min(max_den, int(upper_root))
            
        if lower_bound > upper_bound_d:
            return None

        # To improve search efficiency, try larger denominators first?
        # Usually, greedy (smallest first) works well for finding *a* solution.
        # But to minimize MAX denominator, we might want to explore larger first denominators
        # to allow smaller subsequent denominators?
        # Actually, the objective is to minimize the global max.
        # If we pick a small first denominator, the remainder is larger, requiring larger subsequent denominators?
        # Or smaller?
        # 1/d is large. Remainder is smaller.
        # If remainder is small, we need large denominators to sum to a small number.
        # So picking small d leads to large subsequent denominators.
        # Picking large d (close to den/num) leads to small remainder, which needs... large denominators too?
        # Wait.
        # Target T.
        # Pick d. R = T - 1/d.
        # If d is small, 1/d is large, R is small. To sum to a small R, we need large denominators.
        # If d is large, 1/d is small, R is close to T. To sum to a large R, we need smaller denominators.
        # So picking larger d (closer to the theoretical minimum d=den/num) might allow smaller subsequent denominators?
        # Let's test.
        # T = 5/31 ~ 0.16
        # Min d = 31/5 = 6.2 -> 7.
        # If d=7, 1/7 ~ 0.14. R ~ 0.02. Need large denominators for 0.02.
        # If d=10, 1/10 = 0.1. R = 0.06. Need denominators for 0.06.
        # If d=20, 1/20 = 0.05. R = 0.11. Need denominators for 0.11.
        
        # It seems picking d closer to the lower bound (small d) makes the remainder small, forcing large denominators.
        # Picking d larger makes the remainder larger, potentially allowing smaller denominators.
        # So we should iterate d from UPPER bound DOWN to lower bound?
        # Or maybe we want to balance.
        
        # Let's try iterating from lower_bound to upper_bound_d first.
        # If we find a solution, we update best.
        
        for d in range(lower_bound, upper_bound_d + 1):
            if time.time() - start_time > remaining_time:
                return None
            
            if d in used_set:
                continue
            
            next_num = num * d - den
            next_den = den * d
            
            if next_num < 0:
                continue
                
            if next_num == 0:
                if max_terms == 1:
                    return current_list + [d]
                else:
                    # We have 0 left but terms to fill? No, we just need the sum to match.
                    # The problem doesn't require a specific number of terms, just distinct denominators.
                    # "Return 1..256 distinct denominators".
                    # So if we hit 0, we are done.
                    return current_list + [d]
            
            # Reduce fraction
            g = gcd(next_num, next_den)
            if g > 1:
                next_num //= g
                next_den //= g
            
            # Recursive call
            new_used = used_set | {d}
            res = find_egyptian_fraction(next_num, next_den, d + 1, max_den, max_terms - 1, new_used, current_list + [d], remaining_time)
            if res is not None:
                return res
        
        return None

    # Strategy:
    # We want to minimize the max denominator.
    # We can try different limits for max_den.
    # Start with a known good limit (e.g., 1736) and try to find a solution with a smaller limit.
    # Or, we can run the DFS with a dynamic limit or just a high limit and see what we get.
    # The DFS above finds *a* solution. It doesn't guarantee minimal max denom.
    # However, if we iterate d from lower to upper, we tend to pick small d first.
    # This leads to large subsequent denominators.
    
    # Better approach for minimization:
    # Binary search on the max denominator D?
    # Or, run DFS with a decreasing D.
    
    # Let's try to find a solution with max_den < 1736.
    # We can try D = 1735, 1700, 1500, etc.
    # Or, we can modify the DFS to prioritize solutions with smaller max denominators.
    # But DFS returns the first solution found.
    
    # Let's try a few specific limits.
    # We know 1736 works.
    # Let's try to find a solution with max_den <= 1000.
    # If found, update best.
    # Then try 500, 200, etc.
    
    # To make the DFS smarter, we can pass the target max_den as a constraint.
    # The function find_egyptian_fraction already takes max_den.
    
    # Let's try a range of max_den values.
    # We should also consider that the number of terms might vary.
    # The DFS allows up to max_terms.
    
    # Let's try max_terms up to 10 or 20.
    
    # We will iterate over possible max_den values.
    # To be efficient, we can start from the current best and go down.
    # But checking every integer is slow.
    # Let's try some key values.
    
    # Also, we can try to improve the search order.
    # If we iterate d from upper_bound DOWN to lower_bound, we pick larger d first.
    # This leaves a larger remainder, which might allow smaller subsequent denominators.
    # This could lead to a smaller overall max denominator.
    
    # Let's modify the DFS to iterate in reverse order (larger d first).
    
    def find_egyptian_fraction_rev(num, den, min_den, max_den, max_terms, used_set, current_list, remaining_time):
        if time.time() - start_time > remaining_time:
            return None

        if max_terms == 0:
            if num == 0:
                return current_list
            return None

        if num == 0:
            return current_list

        if num < 0:
            return None

        lower_bound = max(min_den, (den + num - 1) // num)
        
        if num * lower_bound > den * max_terms:
            return None

        a = num
        b = num - den * max_terms
        c = -den
        
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            upper_bound_d = max_den
        else:
            sqrt_disc = math.sqrt(discriminant)
            upper_root = (-b + sqrt_disc) / (2*a)
            upper_bound_d = min(max_den, int(upper_root))
            
        if lower_bound > upper_bound_d:
            return None

        # Iterate from upper_bound_d down to lower_bound
        for d in range(upper_bound_d, lower_bound - 1, -1):
            if time.time() - start_time > remaining_time:
                return None
            
            if d in used_set:
                continue
            
            next_num = num * d - den
            next_den = den * d
            
            if next_num < 0:
                continue
                
            if next_num == 0:
                if max_terms == 1:
                    return current_list + [d]
                else:
                    return current_list + [d]
            
            g = gcd(next_num, next_den)
            if g > 1:
                next_num //= g
                next_den //= g
            
            new_used = used_set | {d}
            res = find_egyptian_fraction_rev(next_num, next_den, d + 1, max_den, max_terms - 1, new_used, current_list + [d], remaining_time)
            if res is not None:
                return res
        
        return None

    # Try to find a better solution
    # We'll try a few max_den limits, starting from a high one and going down.
    # Or just try to find *any* solution with a smaller max_den.
    
    # Let's try max_den = 1000, 500, 200, 100
    # And max_terms = 5, 10, 20
    
    # We can also try the forward search (small d first) as it might find solutions with fewer terms
    # which might have smaller max denominators? No, fewer terms usually means larger denominators.
    
    # Let's try the reverse search (large d first) with various max_den limits.
    
    limits_to_try = [1735, 1500, 1000, 800, 600, 500, 400, 300, 200, 150, 100]
    
    for limit in limits_to_try:
        if time.time() - start_time > time_limit - 5.0:
            break
            
        # Try different max_terms
        for mt in [10, 20, 30]:
            if time.time() - start_time > time_limit - 5.0:
                break
                
            sol = find_egyptian_fraction_rev(5, 31, 1, limit, mt, set(), [], time_limit)
            
            if sol is not None:
                if check_solution(sol):
                    actual_max = max(sol)
                    if actual_max < best_max_denom:
                        best_max_denom = actual_max
                        best_solution = sol
                        # Update limit to search for even smaller
                        # But the loop is fixed. We can break and restart with smaller limits?
                        # For now, just update best.
                        # We can add the new limit to the front of limits_to_try?
                        # Or just continue.
                        pass
    
    # Also try the forward search with some limits, just in case
    limits_to_try_forward = [1735, 1000, 500, 200]
    
    for limit in limits_to_try_forward:
        if time.time() - start_time > time_limit - 5.0:
            break
            
        for mt in [10, 20]:
            if time.time() - start_time > time_limit - 5.0:
                break
                
            sol = find_egyptian_fraction(5, 31, 1, limit, mt, set(), [], time_limit)
            
            if sol is not None:
                if check_solution(sol):
                    actual_max = max(sol)
                    if actual_max < best_max_denom:
                        best_max_denom = actual_max
                        best_solution = sol

    return best_solution
