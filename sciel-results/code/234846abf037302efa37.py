import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    N = 255
    values = list(range(2, 257))
    weights = [1.0/v for v in values]
    
    # Precompute conflict masks
    conflict = [0] * N
    for i in range(N):
        for j in range(i+1, N):
            a, b = values[i], values[j]
            g = math.gcd(a, b)
            if (a * b) // g <= 256:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
                
    # Order nodes by weight descending
    order = sorted(range(N), key=lambda i: weights[i], reverse=True)
    
    best_sum = 0.0
    best_set = []
    
    # Helper to sum weights of bits in mask
    def mask_sum(mask):
        s = 0.0
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            s += weights[idx]
            mask ^= lsb
        return s

    def dfs(k, current_sum, available_mask, sum_avail, current_set):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
            
        if k == N:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = current_set[:]
            return
            
        # Upper bound
        if current_sum + sum_avail <= best_sum:
            return
            
        node = order[k]
        
        if (available_mask & (1 << node)) == 0:
            # Node not available
            dfs(k+1, current_sum, available_mask, sum_avail, current_set)
        else:
            # Node available
            
            # Option 1: Include node
            removed_mask = (1 << node) | (conflict[node] & available_mask)
            
            # Calculate sum of removed weights
            s_removed = 0.0
            m = removed_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                s_removed += weights[idx]
                m ^= lsb
                
            new_sum = current_sum + weights[node]
            new_sum_avail = sum_avail - s_removed
            new_mask = available_mask & ~removed_mask
            
            current_set.append(node)
            dfs(k+1, new_sum, new_mask, new_sum_avail, current_set)
            current_set.pop()
            
            # Option 2: Exclude node
            # Note: We must ensure that if we exclude node, we don't accidentally include it later.
            # Since we process nodes in fixed order `order`, and we are at index k,
            # node will not be considered again.
            # However, `available_mask` still contains `node`.
            # This is fine because we only check `if (available_mask & (1 << node))` for the CURRENT node.
            # For future nodes, they are different.
            # So we can just pass available_mask as is.
            
            dfs(k+1, current_sum, available_mask, sum_avail, current_set)

    initial_mask = (1 << N) - 1
    initial_sum_avail = sum(weights)
    
    dfs(0, 0.0, initial_mask, initial_sum_avail, [])
    
    result = [values[i] for i in best_set]
    return result
