import math

def construct():
    target = math.factorial(12)
    best_max = target
    best_solution = [target]
    
    # Generate all divisors of target
    divisors = set()
    sqrt_t = int(math.isqrt(target))
    for i in range(1, sqrt_t + 1):
        if target % i == 0:
            divisors.add(i)
            divisors.add(target // i)
            
    # Filter divisors > 12 and sort
    valid_divs = sorted([d for d in divisors if d > 12])
    
    def dfs(prod, start_idx, current_factors):
        nonlocal best_max, best_solution
        
        if prod == target:
            # Valid solution found
            # Since we pick in increasing order, the last factor is the largest
            current_max = current_factors[-1]
            if current_max < best_max:
                best_max = current_max
                best_solution = current_factors[:]
            return
        
        # If prod exceeds target, no need to continue
        if prod > target:
            return
            
        # Pruning: 
        # The smallest divisor we can pick next is valid_divs[start_idx].
        # If we pick a divisor d, the new max will be at least d.
        # If d >= best_max, we cannot improve the best_max.
        # So we can break if the current candidate d is >= best_max.
        
        for i in range(start_idx, len(valid_divs)):
            d = valid_divs[i]
            
            # If adding this divisor exceeds target, since divisors are sorted, 
            # all subsequent divisors will also exceed.
            if prod * d > target:
                break
            
            # If this divisor is already >= best_max, any solution using it 
            # will have max >= d >= best_max, so no improvement.
            if d >= best_max:
                break
                
            current_factors.append(d)
            dfs(prod * d, i + 1, current_factors)
            current_factors.pop()

    dfs(1, 0, [])
    
    return best_solution
