import time
from fractions import Fraction

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    target = Fraction(5, 31)
    max_terms = 256
    
    # 1. Get an initial valid solution using Greedy Algorithm
    def greedy(r):
        vals = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            vals.append(d)
            r -= Fraction(1, d)
        return vals
    
    initial_sol = greedy(target)
    if len(initial_sol) > max_terms:
        # Greedy might produce too many terms? 
        # For 5/31, greedy produces:
        # 5/31 -> 1/7 (28/31) -> 28/31 -> 1/2 (24/31) -> 24/31 -> 1/2 (no, distinct)
        # Let's trace:
        # 5/31. ceil(31/5) = 7. 1/7. Rem: 5/31 - 1/7 = (35-31)/217 = 4/217.
        # 4/217. ceil(217/4) = 55. 1/55. Rem: 4/217 - 1/55 = (220-217)/11935 = 3/11935.
        # 3/11935. ceil(11935/3) = 3979. 1/3979. Rem: 3/11935 - 1/3979 = (3*3979 - 11935) / (11935*3979) = (11937-11935)/... = 2/...
        # 2/... -> 1/d ...
        # It usually terminates quickly for small fractions.
        # If it's too long, we might need a different initial bound.
        # But for 5/31, it's short.
        pass
        
    best_max = max(initial_sol)
    best_denominators = initial_sol
    
    def dfs(r, min_d, current_max, depth, path):
        nonlocal best_max, best_denominators, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
        
        if r == 0:
            if current_max < best_max:
                best_max = current_max
                best_denominators = path[:]
            return
        
        if depth >= max_terms:
            return
            
        # Lower bound for next denominator
        d_min = max(min_d, (r.denominator + r.numerator - 1) // r.numerator)
        
        # Upper bound: d must be < best_max
        d_max = best_max - 1
        
        if d_min > d_max:
            return
            
        for d in range(d_min, d_max + 1):
            if time.time() - start_time > time_limit:
                return
                
            r_new = r - Fraction(1, d)
            if r_new < 0:
                continue
                
            new_max = max(current_max, d)
            
            if r_new == 0:
                if new_max < best_max:
                    best_max = new_max
                    best_denominators = path + [d]
            else:
                # Pruning: Next denominator must be < best_max
                next_min_d = max(d + 1, (r_new.denominator + r_new.numerator - 1) // r_new.numerator)
                if next_min_d < best_max:
                    path.append(d)
                    dfs(r_new, next_min_d, new_max, depth + 1, path)
                    path.pop()
                    
    dfs(target, 1, 0, 0, [])
    
    return best_denominators
