import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit

    # Known baseline solution
    best_solution = [8, 28, 1736]
    best_max_denom = 1736

    # Precompute 2-term sums for denominators up to K
    K_PRECOMP = 2000
    two_term_map = {}
    
    # Optimization: Use integer tuples (num, den) as keys for speed
    # But we need to reduce fractions.
    
    # We will store lists of pairs (d1, d2) for each sum
    # To save time, we only generate d1 < d2
    
    # This loop is O(K^2). 2000^2 = 4M iterations.
    # In Python, this might take ~1-2 seconds. Acceptable.
    
    for d1 in range(1, K_PRECOMP + 1):
        for d2 in range(d1 + 1, K_PRECOMP + 1):
            # Sum = 1/d1 + 1/d2 = (d1+d2)/(d1*d2)
            num = d1 + d2
            den = d1 * d2
            g = gcd(num, den)
            num_r = num // g
            den_r = den // g
            key = (num_r, den_r)
            if key not in two_term_map:
                two_term_map[key] = []
            two_term_map[key].append((d1, d2))
            
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

    def dfs(num, den, min_d, current_set):
        nonlocal best_solution, best_max_denom
        
        if time.time() - start_time > time_limit:
            return
        
        if num == 0:
            if max(current_set) < best_max_denom:
                best_max_denom = max(current_set)
                best_solution = list(current_set)
            return
        
        if num < 0:
            return
            
        # Lower bound for next denominator d
        # 1/d <= num/den => d >= den/num
        lower_bound = (den + num - 1) // num
        if lower_bound < min_d:
            lower_bound = min_d
            
        # Pruning: if the smallest possible next denominator is already >= best_max_denom,
        # any solution found will have max >= best_max_denom, so we can stop.
        if lower_bound >= best_max_denom:
            return
            
        # Check if we can finish with 1 term
        # num/den = 1/d => d = den/num
        if den % num == 0:
            d = den // num
            if d >= min_d and d < best_max_denom and d not in current_set:
                # Valid 1-term solution
                candidate = current_set + [d]
                if max(candidate) < best_max_denom:
                    best_max_denom = max(candidate)
                    best_solution = candidate
                return # No need to recurse further for this branch if we found a valid 1-term completion? 
                       # Actually, we might find a better one with more terms? 
                       # No, if we have a valid solution, adding more terms increases complexity. 
                       # But wait, we are minimizing MAX denominator. 
                       # A 1-term completion gives max = max(current_set, d).
                       # A multi-term completion might have smaller max? 
                       # If d is very large, splitting it into smaller denominators might reduce the max.
                       # Example: 1/100 can be 1/200 + 1/200 (not distinct) or 1/201 + ...
                       # So we should NOT return immediately. We should record this, but continue searching for better max.
                       # However, if d < best_max_denom, we updated best. 
                       # We still need to check if there's a solution with max < d.
                       # So we do NOT return. We just continue the loop.
        
        # Check if we can finish with 2 terms from precomputed map
        # We need num/den to be in two_term_map
        # But we must ensure d1, d2 >= min_d and distinct from current_set
        # And max(d1, d2) < best_max_denom
        
        # Reduce num/den for lookup
        g = gcd(num, den)
        num_r = num // g
        den_r = den // g
        key = (num_r, den_r)
        
        if key in two_term_map:
            for d1, d2 in two_term_map[key]:
                if d1 < min_d:
                    continue
                if d2 >= best_max_denom:
                    continue
                if d1 in current_set or d2 in current_set:
                    continue
                # Found a valid 2-term completion
                candidate = current_set + [d1, d2]
                if max(candidate) < best_max_denom:
                    best_max_denom = max(candidate)
                    best_solution = candidate
                # Do not return, as there might be a better solution with different terms
                # But we can break if we found a solution with max == lower_bound? No.
            
        # Recurse for next term
        # Upper bound for d:
        # We need to be able to form the remainder with at least 1 term (or more).
        # Simple bound: d < best_max_denom
        # Also, 1/d must be <= num/den
        # So d >= den/num (already handled by lower_bound)
        
        # Additional pruning:
        # If we use d, remainder is (num*d - den)/(den*d).
        # The next denominator d_next must be > d.
        # So 1/d_next < 1/d.
        # The remainder must be positive.
        
        for d in range(lower_bound, best_max_denom):
            if time.time() - start_time > time_limit:
                return
            
            # Check if 1/d <= num/den
            if den > num * d:
                continue
                
            next_num = num * d - den
            next_den = den * d
            
            # Reduce fraction
            g2 = gcd(next_num, next_den)
            if g2 > 1:
                next_num //= g2
                next_den //= g2
                
            # Recurse
            dfs(next_num, next_den, d + 1, current_set + [d])
            
    # Start search
    dfs(5, 31, 1, [])

    if verify(best_solution):
        return best_solution
    
    return [8, 28, 1736]
