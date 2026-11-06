"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin for 60s limit
    
    # Known valid solution with 10 denominators
    known_solution = [5, 6, 7, 8, 9, 10, 14, 15, 63, 2520]
    
    def max_sum_from(start_d, terms):
        """Calculate max sum of 'terms' distinct reciprocals starting from start_d."""
        s = Fraction(0, 1)
        for i in range(terms):
            s += Fraction(1, start_d + i)
        return s
    
    def find_solution(k, deadline):
        """
        Find k distinct denominators >= 5 summing to 1.
        Returns list of denominators or None.
        """
        target = Fraction(1, 1)
        
        def dfs(start_d, remaining_sum, terms_left, current_solution):
            if time.time() > deadline:
                return None
            
            if terms_left == 0:
                if remaining_sum == 0:
                    return current_solution
                return None
            
            if terms_left == 1:
                # 1/d = remaining_sum => d = 1/remaining_sum
                if remaining_sum == 0:
                    return None
                num, den = remaining_sum.numerator, remaining_sum.denominator
                if num != 1:
                    return None
                d = den
                if d >= start_d and d <= 10000000:
                    return current_solution + [d]
                return None
            
            # Lower bound for d: 1/d <= remaining_sum => d >= 1/remaining_sum
            # Since d is integer, d >= ceil(1/remaining_sum)
            # Also d >= start_d
            inv_remaining = 1 / float(remaining_sum)
            min_d = max(start_d, math.ceil(inv_remaining))
            
            # Upper bound: we need terms_left-1 more terms after d
            # Max sum from d+1 for terms_left-1 terms must be >= remaining_sum - 1/d
            # Approx: (terms_left-1)/(d+1) >= remaining_sum - 1/d
            # For safety, use: d < terms_left / remaining_sum
            max_d_float = float(terms_left) / float(remaining_sum)
            max_d = int(max_d_float) - 1
            max_d = min(max_d, 10000000)
            
            if min_d > max_d:
                return None
            
            for d in range(min_d, max_d + 1):
                if time.time() > deadline:
                    return None
                
                next_remaining = remaining_sum - Fraction(1, d)
                if next_remaining < 0:
                    # 1/d > remaining_sum, try larger d
                    continue
                
                # Check if max possible sum from d+1 with terms_left-1 terms >= next_remaining
                if terms_left - 1 > 0:
                    max_possible = max_sum_from(d + 1, terms_left - 1)
                    if max_possible < next_remaining:
                        # Even taking the largest possible terms won't reach the target
                        # Since d is increasing, max_possible will only decrease, so break
                        break
                
                result = dfs(d + 1, next_remaining, terms_left - 1, current_solution + [d])
                if result is not None:
                    return result
            
            return None
        
        return dfs(5, target, k, [])
    
    # Try k=7 first (theoretical minimum given constraints)
    for k in [7, 8, 9]:
        if time.time() > deadline:
            break
        sol = find_solution(k, deadline)
        if sol is not None:
            # Verify the solution
            total = sum(Fraction(1, d) for d in sol)
            if total == 1 and all(5 <= d <= 10000000 for d in sol) and len(set(sol)) == len(sol):
                return sol
    
    # Fall back to known k=10 solution
    return known_solution
# EVOLVE-BLOCK-END
