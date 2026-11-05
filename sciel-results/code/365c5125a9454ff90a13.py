import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute reciprocal fractions
    recips = [Fraction(1, d) for d in denominators]
    
    # Precompute suffix sums of reciprocals
    # suffix_sum[i] = sum(recips[i] + ... + recips[n-1])
    suffix_sum = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + recips[i]
    
    # Initialize best with a known good solution to aid pruning
    # Indices 0, 1, 5, 41 correspond to 2, 3, 7, 43
    # Sum = 1/2 + 1/3 + 1/7 + 1/43 = 1805/1806
    init_indices = [0, 1, 5, 41]
    init_sum = sum(recips[i] for i in init_indices)
    
    # Verify validity
    if init_sum >= 1:
        # Fallback to a simple valid set if logic changes, but 1805/1806 < 1
        init_indices = [0]
        init_sum = recips[0]
        
    best_sum = init_sum
    best_indices = init_indices[:]
    
    # Branch and Bound
    # State: (index, current_sum, current_indices)
    # We use a stack (LIFO).
    # We push 'Exclude' first, then 'Include', so 'Include' is processed first.
    # This tends to find larger sums earlier, tightening the bound.
    
    stack = [(0, Fraction(0), [])]
    
    while stack:
        if time.time() > deadline:
            break
            
        idx, curr_sum, curr_indices = stack.pop()
        
        if idx == n:
            if curr_sum < 1 and curr_sum > best_sum:
                best_sum = curr_sum
                best_indices = curr_indices[:]
            continue
        
        # Pruning 1: If adding the maximum possible remaining sum doesn't beat best, skip
        # If curr_sum + suffix_sum[idx] <= best_sum, we can't improve.
        if curr_sum + suffix_sum[idx] <= best_sum:
            continue
            
        # Option 1: Exclude current denominator
        # Push first so it's processed second (LIFO)
        stack.append((idx + 1, curr_sum, curr_indices))
        
        # Option 2: Include current denominator
        new_sum = curr_sum + recips[idx]
        if new_sum < 1:
            # Push second so it's processed first
            stack.append((idx + 1, new_sum, curr_indices + [idx]))
        # If new_sum >= 1, we don't push, effectively pruning this branch
        
    return best_indices
