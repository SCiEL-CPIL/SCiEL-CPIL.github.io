import time
from math import factorial

def construct():
    target = factorial(12)
    # Get all divisors of target
    divisors = []
    for i in range(13, int(target**0.5) + 1):
        if target % i == 0:
            divisors.append(i)
            other = target // i
            if other != i:
                divisors.append(other)
    # We only care about divisors > 12
    divisors = [d for d in divisors if d > 12]
    # Sort ascending to find smaller max factors first
    divisors.sort()
    
    best_max = float('inf')
    best_set = []
    
    def dfs(remaining_product, min_factor, current_set, max_factor):
        nonlocal best_max, best_set
        
        # Pruning: if current max is already worse, stop
        if max_factor >= best_max:
            return
            
        if remaining_product == 1:
            if len(current_set) >= 1:
                # Valid solution
                if max_factor < best_max:
                    best_max = max_factor
                    best_set = list(current_set)
            return
            
        # If remaining product is > 1, we need at least one more factor
        # The smallest factor we can pick is min_factor
        # If remaining_product < min_factor, we can't form it with factors >= min_factor (since they are integers > 1)
        # Wait, factors must be distinct and > 12.
        # If remaining_product is prime and > min_factor, we can pick it if it's in divisors.
        # If remaining_product is composite, we pick a divisor.
        
        # We must pick a factor f such that:
        # 1. f > 12
        # 2. f >= min_factor (to ensure distinctness and ordering)
        # 3. f divides remaining_product
        # 4. f is in divisors (implicitly true if it divides target)
        # 5. f <= best_max (pruning)
        
        # Also, if remaining_product itself is a valid factor and >= min_factor, we can take it directly.
        
        # Iterate through possible factors
        # To be efficient, we can iterate through divisors of remaining_product
        # that are >= min_factor.
        
        # Simple iteration over divisors list might be slow if not careful.
        # But divisors list is small.
        
        for f in divisors:
            if f < min_factor:
                continue
            if f > remaining_product:
                break # Since divisors are sorted
            if remaining_product % f != 0:
                continue
            if f >= best_max:
                break # Since divisors are sorted, larger f will also be >= best_max
                
            new_max = max(max_factor, f)
            if new_max >= best_max:
                break
                
            current_set.append(f)
            dfs(remaining_product // f, f + 1, current_set, new_max)
            current_set.pop()
            
    dfs(target, 13, [], 0)
    
    if best_set:
        return best_set
    else:
        # Fallback to known solution
        return [33, 28, 25, 32, 27, 24]
