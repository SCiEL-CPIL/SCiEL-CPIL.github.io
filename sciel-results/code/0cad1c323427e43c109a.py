import math

def construct():
    N = 12
    target = math.factorial(N)
    
    # Prime factorization of 12!
    primes = []
    for _ in range(10): primes.append(2)
    for _ in range(5): primes.append(3)
    for _ in range(2): primes.append(5)
    primes.append(7)
    primes.append(11)
    
    primes.sort(reverse=True)
    
    best_max = target
    best_solution = [target]
    
    def dfs(idx, factors):
        nonlocal best_max, best_solution
        
        if idx == len(primes):
            if len(factors) > 12:
                return
            
            valid = True
            for i in range(len(factors)):
                if factors[i] <= 12:
                    valid = False
                    break
                if i > 0 and factors[i] == factors[i-1]:
                    valid = False
                    break
            
            if valid:
                current_max = factors[-1]
                if current_max < best_max:
                    best_max = current_max
                    best_solution = factors[:]
            return
        
        p = primes[idx]
        
        # Option 1: Assign p to an existing factor
        for i in range(len(factors)):
            new_val = factors[i] * p
            if new_val >= best_max:
                continue
            
            # Create new list without factors[i]
            new_factors = factors[:i] + factors[i+1:]
            
            # Find insertion point for new_val
            lo, hi = 0, len(new_factors)
            while lo < hi:
                mid = (lo + hi) // 2
                if new_factors[mid] < new_val:
                    lo = mid + 1
                else:
                    hi = mid
            
            # Check for duplicates
            if lo < len(new_factors) and new_factors[lo] == new_val:
                continue
            if lo > 0 and new_factors[lo-1] == new_val:
                continue
            
            new_factors.insert(lo, new_val)
            dfs(idx + 1, new_factors)
            
        # Option 2: Create a new factor with value p
        if len(factors) < 12:
            if p < best_max:
                lo, hi = 0, len(factors)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if factors[mid] < p:
                        lo = mid + 1
                    else:
                        hi = mid
                
                # Check for duplicates
                if lo < len(factors) and factors[lo] == p:
                    pass # Duplicate, skip
                elif lo > 0 and factors[lo-1] == p:
                    pass # Duplicate, skip
                else:
                    new_factors = factors[:lo] + [p] + factors[lo:]
                    dfs(idx + 1, new_factors)

    dfs(0, [])
    
    return best_solution
