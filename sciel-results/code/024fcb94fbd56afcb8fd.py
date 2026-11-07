import time
import random
import numpy as np

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 85  # Safety margin before 90s kill

    # 1. Precompute comparability matrix
    # comp_mat[i, j] = 1 if i and j are comparable (and i != j)
    masks = np.arange(total_masks, dtype=np.int32)
    comp_mat = np.zeros((total_masks, total_masks), dtype=np.int8)
    chunk_size = 512
    for start in range(0, total_masks, chunk_size):
        end = min(start + chunk_size, total_masks)
        a_chunk = masks[start:end, np.newaxis]
        b_chunk = masks[np.newaxis, :]
        # a subset b: (a & ~b) == 0
        a_sub_b = (a_chunk & ~b_chunk) == 0
        b_sub_a = (b_chunk & ~a_chunk) == 0
        comp_mat[start:end, :] = (a_sub_b | b_sub_a).astype(np.int8)
    np.fill_diagonal(comp_mat, 0) # Strict subset

    # 2. Greedy Initialization
    selected = np.zeros(m, dtype=np.int32)
    selected_set = set()
    compat = np.zeros(total_masks, dtype=np.int32)
    
    # Start with a random mask
    first_mask = random.randint(0, total_masks - 1)
    selected[0] = first_mask
    selected_set.add(first_mask)
    compat += comp_mat[:, first_mask]
    
    for k in range(1, m):
        # Find mask not in selected_set that maximizes compat
        valid = np.ones(total_masks, dtype=bool)
        for s in selected_set:
            valid[s] = False
        gains = compat * valid
        best_idx = int(np.argmax(gains))
        selected[k] = best_idx
        selected_set.add(best_idx)
        compat += comp_mat[:, best_idx]
        
    # Calculate initial score
    # Sum of comp_mat for all pairs in selected
    # comp_mat is symmetric? Yes.
    # Score = 0.5 * sum(comp_mat[selected, selected])
    score = int(np.sum(comp_mat[np.ix_(selected, selected)]) // 2)
    best_config = selected.copy()
    best_score = score

    # 3. Simulated Annealing
    T = 10.0
    cooling_rate = 0.99995
    
    # Estimate steps per second to adjust cooling? 
    # Just run until time limit.
    
    while time.time() < deadline - 10:
        # Pick random index to swap
        i = random.randint(0, m - 1)
        old_mask = int(selected[i])
        
        # Pick random new mask not in selected_set
        new_mask = random.randint(0, total_masks - 1)
        while new_mask in selected_set:
            new_mask = random.randint(0, total_masks - 1)
        
        # Calculate Delta
        # Delta = compat[new_mask] - compat[old_mask] - comp_mat[new_mask, old_mask]
        delta = int(compat[new_mask]) - int(compat[old_mask]) - int(comp_mat[new_mask, old_mask])
        
        if delta > 0 or random.random() < np.exp(delta / T):
            # Accept
            score += delta
            
            # Update state
            selected_set.remove(old_mask)
            selected_set.add(new_mask)
            selected[i] = new_mask
            
            # Update compat
            compat -= comp_mat[:, old_mask]
            compat += comp_mat[:, new_mask]
            
            if score > best_score:
                best_score = score
                best_config = selected.copy()
            
        # Cool down
        T *= cooling_rate
        if T < 0.1:
            T = 0.1

    # 4. Final Local Search (Steepest Descent)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(m):
            old_mask = int(selected[i])
            loss = int(compat[old_mask])
            
            # Find best new_mask
            valid = np.ones(total_masks, dtype=bool)
            for s in selected_set:
                valid[s] = False
            
            # We want to maximize compat[new] - comp_mat[new, old_mask]
            # Subject to new not in selected_set
            score_arr = compat - comp_mat[:, old_mask]
            score_arr[~valid] = -1
            
            best_idx = int(np.argmax(score_arr))
            best_delta = int(score_arr[best_idx]) - loss
            
            if best_delta > 0:
                new_mask = int(best_idx)
                score += best_delta
                selected_set.remove(old_mask)
                selected_set.add(new_mask)
                selected[i] = new_mask
                compat -= comp_mat[:, old_mask]
                compat += comp_mat[:, new_mask]
                improved = True
                if score > best_score:
                    best_score = score
                    best_config = selected.copy()

    # Convert to lists
    result = []
    for mask in best_config:
        s = [i for i in range(n) if (mask >> i) & 1]
        result.append(s)
    return result
