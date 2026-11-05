import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    seed_elements = [1, 2, 5, 11]
    max_val = 284
    target_size = 20
    
    def compute_initial_sums(elements):
        bits = 0
        n = len(elements)
        for i in range(n):
            for j in range(i, n):
                bits |= 1 << (elements[i] + elements[j])
        return bits

    initial_sums = compute_initial_sums(seed_elements)
    selected_set = set(seed_elements)
    best_solution = list(seed_elements)
    
    # To speed up can_add, we can pre-compute nothing, but the loop over selected is small (max 20).
    
    def dfs(selected_list, selected_set, sums_bits, order):
        nonlocal best_solution
        
        if len(selected_list) > len(best_solution):
            best_solution = list(selected_list)
            
        if len(selected_list) == target_size:
            return selected_list
            
        if time.time() > deadline:
            return None
            
        # Heuristic: We iterate through 'order'
        # 'order' is a list of all candidates 1..284 excluding seeds? 
        # Or we can just iterate 1..284 and check if in selected_set.
        # Iterating 1..284 is 284 checks. 
        # If we pass 'order', we can stop early if we find a solution.
        
        for x in order:
            if x in selected_set:
                continue
                
            # Check collision
            # 1. Check 2*x
            if (sums_bits >> (2 * x)) & 1:
                continue
                
            # 2. Check x + y for all y in selected_list
            valid = True
            new_sums = 1 << (2 * x)
            for y in selected_list:
                s = x + y
                if (sums_bits >> s) & 1:
                    valid = False
                    break
                new_sums |= 1 << s
                
            if not valid:
                continue
                
            # Recurse
            new_selected_list = selected_list + [x]
            new_selected_set = selected_set | {x}
            
            res = dfs(new_selected_list, new_selected_set, sums_bits | new_sums, order)
            if res:
                return res
                
        return None

    all_candidates = [x for x in range(1, max_val + 1) if x not in seed_elements]
    
    # Define different orders
    # 1. Ascending
    # 2. Descending
    # 3. "Center-out": Sort by distance from 142?
    # 4. Random
    
    orders = [
        all_candidates,
        sorted(all_candidates, reverse=True),
        sorted(all_candidates, key=lambda x: abs(x - 142)),
    ]
    
    # Add random orders
    for _ in range(10):
        shuffled = all_candidates.copy()
        random.shuffle(shuffled)
        orders.append(shuffled)
        
    for i, order in enumerate(orders):
        if time.time() > deadline - 2:
            break
            
        # Reset selected state for this attempt
        res = dfs(list(seed_elements), set(seed_elements), initial_sums, order)
        if res:
            return sorted(res)
            
    return sorted(best_solution)
