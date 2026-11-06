import time
import random

def construct():
    """
    Return exactly 12 distinct positive integers <= 1048576 such that all 2^12 = 4096
    subset sums (including the empty subset) are distinct. Minimize the largest element.
    """
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 12
    
    # Fallback: powers of 2
    fallback = [1 << i for i in range(n)]
    best_solution = list(fallback)
    best_max = max(fallback)
    
    def check_valid(elements):
        """Check if all subset sums are distinct."""
        subset_sums = {0}
        for e in elements:
            new_sums = set()
            for s in subset_sums:
                ns = s + e
                if ns in subset_sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            subset_sums |= new_sums
        return True
    
    # Greedy reduction from largest to smallest
    current = list(fallback)
    
    for idx in range(n - 1, -1, -1):
        if time.time() > deadline:
            break
        
        current_val = current[idx]
        for new_val in range(current_val - 1, 0, -1):
            if time.time() > deadline:
                break
            if new_val in current:
                continue
            
            candidate = list(current)
            candidate[idx] = new_val
            
            if check_valid(candidate):
                current = candidate
                break
    
    current_max = max(current)
    if current_max < best_max:
        best_max = current_max
        best_solution = list(current)
    
    # Randomized search to improve further
    random.seed(42)
    current = list(best_solution)
    current_max = best_max
    
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        idx = random.randint(0, n - 1)
        current_val = current[idx]
        if current_val <= 1:
            continue
        
        new_val = random.randint(1, current_val - 1)
        if new_val in current:
            continue
        
        candidate = list(current)
        candidate[idx] = new_val
        
        if check_valid(candidate):
            current = candidate
            new_max = max(current)
            if new_max < current_max:
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_solution = list(current)
    
    return best_solution
