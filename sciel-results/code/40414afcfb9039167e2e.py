from fractions import Fraction
import time

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    recips = [Fraction(1, d) for d in denominators]
    
    # recips is sorted descending by value (1/2 > 1/3 > ...)
    
    # Precompute suffix sums
    suffix_sum = [Fraction(0, 1)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + recips[i]
    
    best_sum = Fraction(0, 1)
    best_set = []
    start_time = time.time()
    timeout = 55.0
    
    def search(idx, current_sum, current_set):
        nonlocal best_sum, best_set
        
        # Timeout check
        if time.time() - start_time > timeout:
            return
            
        # Pruning: Upper bound
        if current_sum + suffix_sum[idx] <= best_sum:
            return
            
        # Base case
        if idx == n:
            if current_sum < 1 and current_sum > best_sum:
                best_sum = current_sum
                best_set = list(current_set)
            return
        
        # Option 1: Include recips[idx]
        new_sum = current_sum + recips[idx]
        if new_sum < 1:
            current_set.append(idx)
            search(idx + 1, new_sum, current_set)
            current_set.pop()
            
        # Option 2: Exclude recips[idx]
        search(idx + 1, current_sum, current_set)
        
    search(0, Fraction(0, 1), [])
    
    if not best_set:
        # Fallback: should not happen, but return something valid
        # Greedy fallback
        curr_sum = Fraction(0,1)
        curr_set = []
        for i in range(n):
            if curr_sum + recips[i] < 1:
                curr_sum += recips[i]
                curr_set.append(i)
        return sorted(curr_set)
        
    return sorted(best_set)
