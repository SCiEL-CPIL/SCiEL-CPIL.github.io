import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = 55.0
    
    N = 256
    # Values 2..256 -> Index 0..254
    # Precompute compatibility
    # compat[i][j] is True if lcm(val_i, val_j) > 256
    # val_i = i + 2
    vals = list(range(2, N + 1))
    n = len(vals)
    
    compat = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if lcm(vals[i], vals[j]) > N:
                compat[i][j] = True
                compat[j][i] = True
    # Self compatibility is not needed for distinct pairs, 
    # but for "is_compatible" we check against others.
    
    def is_compatible(idx, selected_indices):
        # selected_indices is a set or list of indices
        for j in selected_indices:
            if not compat[idx][j]:
                return False
        return True

    # 1. Greedy Initialization
    selected_indices = []
    for i in range(n):
        if is_compatible(i, selected_indices):
            selected_indices.append(i)
            
    best_set_indices = selected_indices[:]
    best_score = sum(1.0 / vals[i] for i in best_set_indices)
    
    # 2. Local Search
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        
        # Convert to set for faster lookup? 
        # selected_indices is small. List is fine.
        
        best_swap_gain = 0
        best_swap = None
        
        # Try removing each x
        # We want to find y < x (val[y] < val[x] -> idx[y] < idx[x] since vals is sorted)
        # that is compatible with S \ {x}
        
        for idx_x in selected_indices:
            if (time.time() - start_time) > deadline:
                break
                
            # S without x
            S_minus_x = [idx for idx in selected_indices if idx != idx_x]
            
            # Look for y < idx_x
            # Since vals are sorted 2..256, idx corresponds to value order.
            # Smaller index = smaller value = higher weight.
            # We want the smallest index y (best weight) such that y < idx_x 
            # and y not in S_minus_x and compatible with S_minus_x.
            
            best_y = None
            for idx_y in range(0, idx_x):
                if idx_y in S_minus_x:
                    continue
                if is_compatible(idx_y, S_minus_x):
                    best_y = idx_y
                    break # First one is smallest index -> smallest value -> max weight
            
            if best_y is not None:
                gain = 1.0 / vals[best_y] - 1.0 / vals[idx_x]
                if gain > best_swap_gain:
                    best_swap_gain = gain
                    best_swap = (idx_x, best_y)
        
        if best_swap:
            idx_out, idx_in = best_swap
            new_selected = [idx for idx in selected_indices if idx != idx_out]
            new_selected.append(idx_in)
            selected_indices = new_selected
            
            current_score = sum(1.0 / vals[i] for i in selected_indices)
            if current_score > best_score:
                best_score = current_score
                best_set_indices = selected_indices[:]
            improved = True
            
        # Try adding any compatible element
        if (time.time() - start_time) < deadline:
            for idx_y in range(n):
                if idx_y in selected_indices:
                    continue
                if is_compatible(idx_y, selected_indices):
                    selected_indices.append(idx_y)
                    current_score = sum(1.0 / vals[i] for i in selected_indices)
                    if current_score > best_score:
                        best_score = current_score
                        best_set_indices = selected_indices[:]
                    improved = True
                    break # Add one, then re-evaluate swaps
    
    # Convert indices back to values
    return [vals[i] for i in best_set_indices]
