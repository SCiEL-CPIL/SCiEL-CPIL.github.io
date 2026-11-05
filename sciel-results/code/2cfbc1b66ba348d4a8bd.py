from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # We want to minimize the number of terms.
    # Try k=5, then k=6, etc.
    # Theoretical min is likely 5 or 6.
    
    def dfs(rem, min_d, depth, max_depth, path):
        if time.time() > deadline:
            return None
        
        if depth == max_depth:
            if rem == 0:
                return path[:]
            return None
        
        if rem == 0:
            if depth < max_depth: # We found a shorter solution? 
                # Actually we are searching for fixed max_depth first
                return path[:]
            return None
        
        if rem <= 0:
            return None
        
        # Lower bound on next denominator
        # 1/d < rem  => d > 1/rem
        # Also d > min_d
        # Also we need to be able to fill rem with (max_depth - depth) terms.
        # Max sum with remaining terms is sum of 1/min_d + 1/(min_d+1) ... 
        # But a simpler bound: 
        # If rem > (max_depth - depth) / min_d, it's impossible.
        
        max_terms_left = max_depth - depth
        if rem > Fraction(max_terms_left, min_d):
            return None
            
        # Upper bound on next denominator
        # 1/d < rem => d > 1/rem
        # Also 1/d >= rem / max_terms_left (if we use max_terms_left terms of 1/d)
        # So d <= max_terms_left / rem
        
        d_lower = max(min_d, int(1 / rem) + 1)
        d_upper = int(max_terms_left / rem)
        
        if d_upper > 10000000:
            d_upper = 10000000
            
        for d in range(d_lower, d_upper + 1):
            if d in path:
                continue
            new_rem = rem - Fraction(1, d)
            if new_rem < 0:
                break
            # Optimization: if new_rem is 0 and depth < max_depth, we can return 
            # but we are fixing max_depth. If we find a solution with fewer terms, 
            # we should have found it in a previous iteration of the outer loop.
            
            res = dfs(new_rem, d + 1, depth + 1, max_depth, path + [d])
            if res:
                return res
        return None

    # Try k=5
    for k in range(5, 10):
        if time.time() > deadline:
            break
        path = []
        res = dfs(Fraction(1, 1), 5, 0, k, path)
        if res:
            return res

    # Fallback: return a known valid solution (e.g. 6 terms)
    # 1/5 + 1/6 + 1/7 + 1/42 + 1/3 + ... no
    # Just return a generic valid one if search fails (shouldn't happen)
    # A known 6-term solution: 1/5+1/6+1/7+1/10+1/14+1/35 = 1
    # Check: 1/5+1/10 = 3/10. 1/6+1/14+1/35?
    # 1/6+1/14 = (7+3)/42 = 10/42 = 5/21.
    # 5/21 + 1/35 = (25+3)/105 = 28/105 = 4/15.
    # 3/10 + 4/15 = 9/30 + 8/30 = 17/30 != 1.
    
    # Let's use the current program's logic as a fallback if search fails
    # But the search should find one.
    
    # Fallback to current program's result if search fails
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    raise RuntimeError("no solution found")
