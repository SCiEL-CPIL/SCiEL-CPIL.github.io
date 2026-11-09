import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 255
    numbers = list(range(2, 257))
    
    # Build compatibility graph
    # adj[i] has bit j set if numbers[i] and numbers[j] are compatible
    adj = [0] * n
    for i in range(n):
        ai = numbers[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            aj = numbers[j]
            if lcm(ai, aj) > 256:
                mask |= (1 << j)
        adj[i] = mask
        
    weights = [1.0 / x for x in numbers]
    
    # Reorder vertices by weight descending
    # Index 0 will be the heaviest
    order = sorted(range(n), key=lambda i: weights[i], reverse=True)
    new_index = [0] * n
    for new_i, old_i in enumerate(order):
        new_index[old_i] = new_i
        
    new_weights = [0.0] * n
    new_adj = [0] * n
    for old_i in range(n):
        new_i = new_index[old_i]
        new_weights[new_i] = weights[old_i]
        old_mask = adj[old_i]
        new_mask = 0
        for old_j in range(n):
            if (old_mask >> old_j) & 1:
                new_j = new_index[old_j]
                new_mask |= (1 << new_j)
        new_adj[new_i] = new_mask
        
    adj = new_adj
    weights = new_weights
    
    best_weight = 0.0
    best_set_mask = 0
    
    def expand(clique_mask, candidates_mask, current_weight):
        nonlocal best_weight, best_set_mask
        
        if time.time() > deadline:
            return
            
        if candidates_mask == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set_mask = clique_mask
            return
            
        # Bound: current_weight + sum(weights of candidates)
        # Calculate sum of weights in candidates_mask
        rem_mask = candidates_mask
        rem_sum = 0.0
        while rem_mask:
            b = (rem_mask & -rem_mask).bit_length() - 1
            rem_sum += weights[b]
            rem_mask &= ~(1 << b)
            
        if current_weight + rem_sum <= best_weight:
            return
            
        # Branch on heaviest candidate (lowest index)
        c_mask = candidates_mask
        while c_mask:
            if time.time() > deadline:
                return
            
            v = (c_mask & -c_mask).bit_length() - 1
            
            # Include v
            new_clique = clique_mask | (1 << v)
            new_candidates = candidates_mask & adj[v]
            # Note: new_candidates does not include v because adj[v] doesn't have bit v.
            # It also doesn't include vertices not in candidates_mask.
            
            expand(new_clique, new_candidates, current_weight + weights[v])
            
            # Exclude v
            candidates_mask &= ~(1 << v)
            
            # Update bound check for the remaining candidates in this loop
            # We can re-calculate rem_sum or just check if the max possible is too low
            # Re-calculating is safer
            rem_mask = candidates_mask
            rem_sum = 0.0
            while rem_mask:
                b = (rem_mask & -rem_mask).bit_length() - 1
                rem_sum += weights[b]
                rem_mask &= ~(1 << b)
                
            if current_weight + rem_sum <= best_weight:
                break

    initial_candidates = (1 << n) - 1
    expand(0, initial_candidates, 0.0)
    
    result = []
    for i in range(n):
        if (best_set_mask >> i) & 1:
            result.append(numbers[i])
            
    if not result:
        result = [2]
        
    return result
