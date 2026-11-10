import math
import time

def construct():
    # Target product
    target = math.factorial(12)
    
    # Generate all divisors of target
    divisors = []
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # We can just iterate up to sqrt(target) or use the prime factorization
    # For 12!, iterating up to sqrt is fine (sqrt(4.8e8) ~ 21900)
    # Or simpler:
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
            
    # Filter divisors > 12
    valid_divisors = [d for d in divisors if d > 12]
    valid_divisors.sort()
    
    best_solution = []
    best_max = float('inf')
    
    # Time guard
    start_time = time.time()
    
    def dfs(remaining, min_idx, current_factors):
        nonlocal best_solution, best_max
        
        # Time check
        if time.time() - start_time > 50:
            return
            
        if remaining == 1:
            if current_factors:
                max_f = max(current_factors)
                if max_f < best_max:
                    best_max = max_f
                    best_solution = list(current_factors)
            return
        
        if len(current_factors) == 12:
            return
            
        # Iterate through possible next factors
        # We start from min_idx to ensure distinctness and increasing order
        for i in range(min_idx, len(valid_divisors)):
            d = valid_divisors[i]
            
            # Pruning: if d is already >= best_max, no need to continue
            # because d will be part of the solution, so max will be at least d
            if d >= best_max:
                break
                
            if remaining % d != 0:
                continue
                
            new_remaining = remaining // d
            
            # Pruning: if new_remaining > 1, we need at least one more factor.
            # The next factor must be > d, so at least valid_divisors[i+1].
            # If new_remaining < valid_divisors[i+1], it's impossible.
            if new_remaining > 1:
                if i + 1 < len(valid_divisors):
                    next_min = valid_divisors[i+1]
                    if new_remaining < next_min:
                        continue
                else:
                    # No more divisors available, but we need to form new_remaining > 1
                    continue
                    
            current_factors.append(d)
            dfs(new_remaining, i + 1, current_factors)
            current_factors.pop()
            
    dfs(target, 0, [])
    
    if best_solution:
        return best_solution
    else:
        # Fallback, though a solution is guaranteed to exist (e.g., [12!])
        # Wait, [12!] has 1 factor, which is > 12.
        # So best_solution should never be empty.
        return [target]
