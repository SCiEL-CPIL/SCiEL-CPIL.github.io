import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    N = 511
    weights = [1.0 / (512 - i) for i in range(1, N + 1)]
    
    # Sort indices by weight descending
    # indices are 1..511
    sorted_indices = sorted(range(1, N + 1), key=lambda x: weights[x-1], reverse=True)
    
    # Map sorted position to original value
    vals = sorted_indices
    M = len(vals)
    
    # Precompute conflict masks for each position in sorted order
    conflict_mask = [0] * M
    
    for i in range(M):
        v_i = vals[i]
        mask = 0
        for j in range(M):
            if i == j:
                continue
            v_j = vals[j]
            if gcd(v_i, v_j) > 1:
                mask |= (1 << j)
        conflict_mask[i] = mask
        
    # Precompute suffix sums of weights for O(1) upper bound
    suffix_sum = [0.0] * (M + 1)
    for i in range(M - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[vals[i] - 1]
        
    # Precompute forward masks
    full_mask = (1 << M) - 1
    forward_masks = [0] * (M + 1)
    for i in range(M + 1):
        # Bits i to M-1 set
        if i == M:
            forward_masks[i] = 0
        else:
            forward_masks[i] = full_mask ^ ((1 << i) - 1)
            
    best_sum = 0.0
    best_set = []
    
    def dfs(idx, current_sum, selected, available_mask):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
            
        # Upper bound: current_sum + sum of weights of all vertices from idx to M-1
        # This is an optimistic bound ignoring conflicts among remaining available vertices.
        bound = current_sum + suffix_sum[idx]
        
        if bound <= best_sum:
            return
            
        if idx >= M:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = list(selected)
            return
            
        if (available_mask & (1 << idx)) != 0:
            # Include idx
            new_selected = selected + [vals[idx]]
            next_mask_include = (available_mask & ~(1 << idx) & ~conflict_mask[idx]) & forward_masks[idx + 1]
            dfs(idx + 1, current_sum + weights[vals[idx] - 1], new_selected, next_mask_include)
            
            # Exclude idx
            next_mask_exclude = (available_mask & ~(1 << idx)) & forward_masks[idx + 1]
            dfs(idx + 1, current_sum, selected, next_mask_exclude)
            
        else:
            # Must exclude
            next_mask_exclude = (available_mask & ~(1 << idx)) & forward_masks[idx + 1]
            dfs(idx + 1, current_sum, selected, next_mask_exclude)

    dfs(0, 0.0, [], full_mask)
    
    # Local Search Polish
    if best_set:
        current_set = set(best_set)
        current_score = best_sum
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding
            for a in range(1, N + 1):
                if a in current_set:
                    continue
                is_coprime = True
                for b in current_set:
                    if gcd(a, b) > 1:
                        is_coprime = False
                        break
                if is_coprime:
                    new_score = current_score + weights[a - 1]
                    if new_score > current_score:
                        current_set.add(a)
                        current_score = new_score
                        improved = True
                        break
            # Try swapping
            if not improved:
                for a in list(current_set):
                    for b in range(1, N + 1):
                        if b in current_set:
                            continue
                        is_coprime = True
                        for c in current_set:
                            if c == a:
                                continue
                            if gcd(b, c) > 1:
                                is_coprime = False
                                break
                        if is_coprime:
                            new_score = current_score - weights[a - 1] + weights[b - 1]
                            if new_score > current_score:
                                current_set.remove(a)
                                current_set.add(b)
                                current_score = new_score
                                improved = True
                                break
                    if improved:
                        break
        if current_score > best_sum:
            best_sum = current_score
            best_set = list(current_set)

    return best_set
