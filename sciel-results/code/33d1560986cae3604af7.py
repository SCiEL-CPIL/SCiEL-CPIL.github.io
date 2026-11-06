import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin for the 90s limit
    
    # Known best solution: 1/8 + 1/28 + 1/1736 = 5/31
    best_solution = [8, 28, 1736]
    best_max_denom = 1736
    
    # We want to find if a solution exists with max denom < best_max_denom.
    # If find_solution_with_limit(K) returns None, no solution exists for any limit < K.
    # So we can binary search or linearly decrease the limit.
    
    def find_solution_with_limit(limit):
        """
        Finds any solution for 5/31 using distinct denominators in [1, limit].
        Returns the list of denominators or None.
        Uses DFS with pruning based on exact rational remainder.
        """
        # State: (num, den, min_d, current_set)
        # Target: 5/31
        # We track the remainder as a reduced fraction num/den.
        
        # Pre-compute harmonic-like bounds for pruning? 
        # Actually, a simple bound: if remaining terms can't sum to num/den, prune.
        # Max sum of reciprocals of distinct integers >= min_d is approx ln(inf/min_d) + gamma,
        # but we have a finite limit.
        # A tighter bound: sum_{i=1}^{limit - min_d + 1} 1/(min_d + i - 1)
        # This is expensive to compute on the fly.
        
        # Alternative pruning: 
        # 1. If num/den == 0, return current_set.
        # 2. If num/den < 0, return None.
        # 3. Lower bound for next d: d >= ceil(den/num).
        # 4. Upper bound for next d: d <= limit.
        # 5. If we have too few terms left to reach the sum? 
        #    Min sum of k distinct reciprocals starting from min_d is roughly 
        #    sum_{i=0}^{k-1} 1/(min_d+i). This is small.
        #    Actually, the main constraint is that we can't exceed the sum.
        #    The sum of ALL available reciprocals from min_d to limit is H(limit) - H(min_d-1).
        #    If num/den > H(limit) - H(min_d-1), it's impossible.
        
        # Let's implement a simple DFS.
        # Optimization: Try smaller denominators first? Or larger?
        # Smaller denominators consume more of the sum, leading to smaller remainders.
        # This tends to find solutions faster if they exist.
        
        def dfs(num, den, min_d, current_set, depth):
            if time.time() > deadline:
                return None
            
            if num == 0:
                return current_set
            
            if num < 0:
                return None
            
            # Lower bound for d: 1/d <= num/den => d >= den/num
            lower_bound = (den + num - 1) // num
            if lower_bound < min_d:
                lower_bound = min_d
            
            if lower_bound > limit:
                return None
            
            # Pruning: Can the remaining denominators sum to num/den?
            # Max possible sum = sum_{d=lower_bound}^{limit} 1/d
            # Approximation: ln(limit) - ln(lower_bound-1)
            # If num/den > max_possible_sum, prune.
            # This is a loose bound but cheap.
            # Let's use a tighter bound for small depths?
            # For now, rely on the loop.
            
            # Iterate d from lower_bound to limit
            # Heuristic: Try d in a specific order?
            # Standard greedy tries smallest d first.
            # Let's try smallest d first.
            
            for d in range(lower_bound, limit + 1):
                if time.time() > deadline:
                    return None
                
                # Check if 1/d <= num/den
                # den <= num * d
                if den > num * d:
                    # This shouldn't happen if d >= ceil(den/num)
                    # But due to integer arithmetic, let's be safe.
                    # Actually, if d >= ceil(den/num), then d >= den/num, so num*d >= den.
                    # So this check is redundant if lower_bound is correct.
                    pass
                
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
        # Note: min_denom should be 1, but denominators must be distinct and >= 1.
        # However, 1/1 = 1 > 5/31, so d=1 is impossible.
        # The lower bound calculation will handle this: ceil(31/5) = 7.
        return dfs(5, 31, 1, [], 0)

    # Main Search Loop
    # We know a solution exists with max 1736.
    # Check if a solution exists with max < 1736.
    # We can binary search the limit K in [1, 1735].
    # Or just check K = 1735. If it fails, is 1736 optimal?
    # Wait, if find_solution_with_limit(1735) returns None, it means no solution with max <= 1735.
    # Since we have a solution with max 1736, then 1736 is the minimum possible max.
    # So we only need to check K = 1735.
    
    # But what if there's a solution with max 1000 that the DFS missed due to timeout or ordering?
    # The DFS is exhaustive for a given limit. If it returns None, it's proven no solution exists for that limit.
    # The only risk is timeout.
    
    # Let's try K = 1735.
    current_limit = best_max_denom - 1
    
    if time.time() < deadline - 1.0:
        sol = find_solution_with_limit(current_limit)
        
        if sol is not None:
            # Found a solution with max <= current_limit
            actual_max = max(sol)
            if actual_max < best_max_denom:
                best_max_denom = actual_max
                best_solution = sol
                
            # Now try to find a solution with even smaller max
            # Set new limit to actual_max - 1
            current_limit = actual_max - 1
            
            # Continue searching downwards?
            # We can loop while current_limit >= 1
            while current_limit >= 1 and time.time() < deadline - 1.0:
                sol = find_solution_with_limit(current_limit)
                if sol is not None:
                    actual_max = max(sol)
                    if actual_max < best_max_denom:
                        best_max_denom = actual_max
                        best_solution = sol
                    current_limit = actual_max - 1
                else:
                    # No solution found for this limit.
                    # This implies no solution exists for any limit < current_limit.
                    # So best_solution is optimal.
                    break
        else:
            # No solution found for limit 1735.
            # This implies 1736 is the optimal max denominator.
            pass
            
    return best_solution
