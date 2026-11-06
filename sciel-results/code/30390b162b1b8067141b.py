import math
from functools import lru_cache

def construct():
    N = 12
    target = math.factorial(N)
    
    # 1. Generate all divisors of target
    divisors = []
    i = 1
    while i * i <= target:
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
        i += 1
    
    # Filter divisors > 12
    valid_divisors = [d for d in divisors if d > 12]
    valid_divisors.sort()
    
    best_max = target
    best_solution = [target]
    
    def dfs(rem_prod, min_next, current_max, count, path):
        nonlocal best_max, best_solution
        
        if rem_prod == 1:
            if current_max < best_max:
                best_max = current_max
                best_solution = path[:]
            return
        
        if count >= 12:
            return
            
        # Pruning 1: Check if rem_prod can be formed by remaining slots
        slots_left = 12 - count
        # We need to pick 'slots_left' distinct integers from [min_next, best_max)
        # such that their product is rem_prod.
        
        # Min product check:
        # The smallest possible product of 'slots_left' distinct integers >= min_next
        # is min_next * (min_next + 1) * ... * (min_next + slots_left - 1)
        # If rem_prod is smaller than this, it's impossible.
        min_prod = 1
        for i in range(slots_left):
            min_prod *= (min_next + i)
            # Early break if min_prod exceeds rem_prod
            if min_prod > rem_prod:
                return
        
        # Max product check:
        # The largest possible product of 'slots_left' distinct integers < best_max
        # is (best_max - 1) * (best_max - 2) * ... * (best_max - slots_left)
        # If rem_prod is larger than this, it's impossible.
        # Note: We must ensure these integers are >= min_next.
        # If best_max - slots_left < min_next, then we don't have enough distinct integers < best_max that are >= min_next.
        if best_max - slots_left < min_next:
            return
            
        max_prod = 1
        for i in range(slots_left):
            factor = best_max - 1 - i
            if factor < min_next:
                break
            max_prod *= factor
            # Early break if max_prod exceeds rem_prod? No, we need max_prod >= rem_prod.
            # If max_prod is already > rem_prod, we can't break early for the "impossible" check, 
            # but we can stop calculating if it gets huge? 
            # Actually, if max_prod < rem_prod, we return.
        if max_prod < rem_prod:
            return

        for d in valid_divisors:
            if d < min_next:
                continue
            if d >= best_max:
                break
            
            if rem_prod % d != 0:
                continue
            
            new_rem = rem_prod // d
            new_max = max(current_max, d)
            
            # If new_max >= best_max, no improvement possible from this branch?
            # Yes, because max will be at least new_max.
            if new_max >= best_max:
                continue
                
            path.append(d)
            dfs(new_rem, d + 1, new_max, count + 1, path)
            path.pop()
            
    dfs(target, 13, 1, 0, [])
    
    return best_solution
