import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit

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
        s = sum(Fraction(1, d) for d in solution)
        return s == Fraction(5, 31)

    def find_solution_with_limit(limit):
        """
        Finds any solution for 5/31 using distinct denominators in [1, limit].
        Returns the list of denominators or None.
        Uses integer arithmetic: state is (num, den) representing num/den.
        """
        # Target: 5/31
        target_num = 5
        target_den = 31

        def dfs(num, den, min_d, current_set, depth):
            if time.time() - start_time > time_limit:
                return None
            
            if num == 0:
                return current_set
            
            if num < 0:
                return None
            
            # Lower bound for next denominator d:
            # 1/d <= num/den => d >= den/num
            # Also d >= min_d
            lower_bound = (den + num - 1) // num
            if lower_bound < min_d:
                lower_bound = min_d
            
            if lower_bound > limit:
                return None
            
            # Pruning: Check if it's possible to sum to num/den using denominators > min_d-1 and <= limit
            # Max possible sum is sum_{d=lower_bound}^limit 1/d
            # For small remaining terms, we can just iterate.
            # A strong prune: if 1/lower_bound > num/den, we can't use lower_bound? 
            # No, lower_bound ensures 1/lower_bound <= num/den.
            
            # Iterate possible denominators
            # Heuristic: Try smaller d first? 
            # If we try smaller d, we reduce the remainder faster, potentially leading to smaller depth.
            # However, we want to find *any* solution.
            
            for d in range(lower_bound, limit + 1):
                if time.time() - start_time > time_limit:
                    return None
                
                # 1/d <= num/den <=> den <= num * d
                if den > num * d:
                    continue
                
                next_num = num * d - den
                next_den = den * d
                
                # Reduce fraction
                g = gcd(next_num, next_den)
                if g > 1:
                    next_num //= g
                    next_den //= g
                
                # Recurse
                res = dfs(next_num, next_den, d + 1, current_set + [d], depth + 1)
                if res is not None:
                    return res
                    
            return None
        
        # Start with remainder 5/31, min_denom 1
        return dfs(target_num, target_den, 1, [], 0)

    # Verify baseline
    if not verify(best_solution):
        # Fallback: greedy construction
        r = Fraction(5, 31)
        solution = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d > 10000000:
                break
            solution.append(d)
            r -= Fraction(1, d)
        if solution and verify(solution):
            best_solution = solution
            best_max_denom = max(solution)
        else:
            return [8, 28, 1736]

    # Linear descent from best_max_denom - 1 down to 1
    # If a solution exists for K, we try K-1.
    # If no solution exists for K, no solution exists for K-1 (monotonicity of feasibility).
    # So if DFS *exhaustively* fails for K, we can stop.
    # However, DFS might fail due to time limit. We assume for small K, it is exhaustive.
    
    current_limit = best_max_denom - 1
    
    while current_limit >= 1:
        if time.time() - start_time > time_limit - 2.0:
            break
        
        sol = find_solution_with_limit(current_limit)
        
        if sol is not None:
            # Found a solution with max <= current_limit
            actual_max = max(sol)
            if actual_max < best_max_denom:
                best_max_denom = actual_max
                best_solution = sol
            
            # Try even smaller limit
            current_limit = actual_max - 1
        else:
            # No solution found for this limit.
            # Assuming DFS was exhaustive (which it should be for small K),
            # this means no solution exists for this limit or lower.
            break
        
        # Safety: if we've gone too low, stop
        if current_limit < 7:  # 1/7 is the smallest possible first term for 5/31
            break

    # Final verification
    if verify(best_solution):
        return best_solution
    
    return [8, 28, 1736]
