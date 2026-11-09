import math
from functools import lru_cache

def construct():
    # Calculate 12!
    target = math.factorial(12)
    
    # Generate all divisors of target
    # Target = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # We can just iterate up to sqrt(target) or generate from prime factors.
    # Generating from prime factors is safer and faster for exactness.
    
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divisors = []
        power = 1
        for _ in range(e + 1):
            for d in divisors:
                new_divisors.append(d * power)
            power *= p
        divisors = new_divisors
    
    # Filter divisors > 12
    valid_divisors = [d for d in divisors if d > 12]
    valid_divisors.sort()
    
    best_set = None
    best_max = float('inf')
    
    # DFS to find all valid decompositions
    # state: index in valid_divisors, current product, current list of factors
    # We use a recursive function
    
    def dfs(start_idx, current_prod, current_factors):
        nonlocal best_set, best_max
        
        # Pruning: if current product exceeds target, stop
        if current_prod > target:
            return
            
        # If we have reached the target, check if this is a better solution
        if current_prod == target:
            if current_factors: # Must have at least 1 factor? Task says 1..12.
                max_factor = max(current_factors)
                if max_factor < best_max:
                    best_max = max_factor
                    best_set = current_factors[:]
            return
            
        # If we have used 12 factors and haven't reached target, stop
        if len(current_factors) >= 12:
            return
            
        # Iterate through possible next factors
        for i in range(start_idx, len(valid_divisors)):
            d = valid_divisors[i]
            
            # Pruning: if current_prod * d > target, we can break because divisors are sorted
            if current_prod * d > target:
                break
                
            # Check if d divides the remaining part of the target
            # i.e., target / current_prod must be divisible by d
            if target % (current_prod * d) == 0:
                dfs(i + 1, current_prod * d, current_factors + [d])
                
    dfs(0, 1, [])
    
    if best_set is None:
        # Fallback to the trivial solution if no decomposition found (should not happen)
        return [target]
        
    return best_set
