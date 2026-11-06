import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    denominators = list(range(2, 49))
    n = len(denominators)

    # Calculate LCM
    L = 1
    for d in denominators:
        L = L * d // gcd(L, d)
        
    weights = [L // d for d in denominators]
    
    # Sort denominators in descending order (weights in descending order)
    # We want to explore larger weights first to find good candidates quickly
    indices_sorted = list(range(n))
    indices_sorted.sort(key=lambda i: weights[i], reverse=True)
    
    # Reorder weights for search
    search_weights = [weights[i] for i in indices_sorted]
    
    # Precompute suffix sums of search_weights
    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + search_weights[i]

    best_sum = 0
    best_indices = []
    
    # Branch and bound DFS
    # state: (idx, current_sum, current_indices)
    # We use recursion with manual stack to avoid recursion limit issues? 
    # n=47, recursion depth is fine in Python if limit is raised, but iterative is safer.
    # Let's use iterative DFS with explicit stack.
    
    # Stack items: (idx, current_sum, indices_list)
    # To save memory, we can store indices as a list and append/pop.
    
    stack = [(0, 0, [])]
    
    while stack:
        if time.time() > deadline:
            break
            
        idx, current_sum, current_indices = stack.pop()
        
        # Pruning:
        # 1. If current_sum >= L, this branch is invalid (sum must be strictly below L).
        #    Actually, if current_sum >= L, we can't add more. And we can't remove (DFS order).
        #    So if current_sum >= L, this path is dead.
        #    Wait, we only push if sum < L? 
        #    If we push (idx, sum, inds), it implies sum < L.
        
        # 2. If idx == n, we have a complete subset.
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_indices = current_indices
            continue
            
        # 3. Pruning based on upper bound:
        #    Max possible sum from this state is current_sum + suffix_sums[idx]
        #    If this max <= best_sum, no improvement possible.
        if current_sum + suffix_sums[idx] <= best_sum:
            continue
            
        # Option 1: Exclude current item (search_weights[idx])
        # Go to idx+1 with same sum
        stack.append((idx + 1, current_sum, current_indices))
        
        # Option 2: Include current item
        new_sum = current_sum + search_weights[idx]
        if new_sum < L:
            # Add to stack. 
            # To explore "include" branches first (to find high sums quickly), 
            # we should push "exclude" first, then "include", so "include" is popped next.
            # But we want to find good best_sum early to prune.
            # So let's push exclude, then include.
            stack.append((idx + 1, new_sum, current_indices + [indices_sorted[idx]]))
            
    if best_indices:
        # Verify and return original indices
        val = sum(weights[i] for i in best_indices)
        if val < L:
            return best_indices
    else:
        # Fallback
        return [0, 1, 5, 41]
