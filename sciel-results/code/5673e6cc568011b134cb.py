# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 55
    target = Fraction(5, 31)
    
    best_max = 248
    best_solution = [8, 31, 248]
    
    # We want to find a solution with max denominator < best_max
    # DFS state: (num, den, last_d, depth, path)
    # num/den is the remaining fraction
    
    max_depth = 20
    
    def dfs(num, den, last_d, depth, path):
        nonlocal best_max, best_solution
        
        if num == 0:
            # Found a solution
            if path and max(path) < best_max:
                best_max = max(path)
                best_solution = path[:]
            return True
        
        if depth >= max_depth:
            return False
            
        # Pruning: 
        # 1. We need num/den = sum of 1/d_i for d_i > last_d
        # The maximum sum we can achieve with (max_depth - depth) terms 
        # using denominators > last_d is bounded by the sum of the first (max_depth - depth) 
        # reciprocals of integers > last_d.
        # If num/den > this max sum, we can't reach it.
        
        remaining_terms = max_depth - depth
        max_possible_sum = Fraction(0, 1)
        d = last_d + 1
        for i in range(remaining_terms):
            max_possible_sum += Fraction(1, d)
            d += 1
            if time.time() > deadline:
                return False
        
        if Fraction(num, den) > max_possible_sum:
            return False
            
        # 2. Also, we must be able to form num/den with denominators < best_max?
        # Not strictly required for correctness, but helps if we restrict search space.
        # However, we are searching for ANY solution with max < best_max.
        # So we should restrict d < best_max.
        
        # Lower bound for next d:
        # 1/d < num/den  => d > den/num
        d_min = den // num + 1
        if d_min <= last_d:
            d_min = last_d + 1
            
        # Upper bound for next d:
        # We want max denominator < best_max, so d < best_max.
        # Also, if 1/d >= num/den, we overshoot.
        # So d > den/num.
        # We iterate d from d_min to best_max - 1.
        
        for d in range(d_min, best_max):
            if time.time() > deadline:
                return False
                
            # Check if 1/d < num/den
            # den < num * d
            if num * d <= den:
                continue
                
            new_num = num * d - den
            new_den = den * d
            
            # Optional: simplify fraction?
            # gcd = math.gcd(new_num, new_den)
            # new_num //= gcd
            # new_den //= gcd
            
            path.append(d)
            dfs(new_num, new_den, d, depth + 1, path)
            path.pop()
            
            # If we found a better solution, best_max is updated.
            # We can break if d >= best_max? No, best_max might decrease.
            # But if best_max decreases, the loop range `range(d_min, best_max)` 
            # is evaluated at the start of the loop? 
            # In Python, range is evaluated once. 
            # So if best_max changes inside, the loop continues to the old upper bound.
            # This is fine, but we might check `if d >= best_max: break` inside.
            if d >= best_max:
                break
                
    dfs(5, 31, 0, 0, [])
    
    return best_solution
# EVOLVE-BLOCK-END
