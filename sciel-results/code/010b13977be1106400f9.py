import numpy as np
import time
import random

# Pre-compute indices for k up to 40
TRIU_INDICES = {}
for k in range(41):
    if k == 0:
        TRIU_INDICES[k] = (np.array([], dtype=np.int32), np.array([], dtype=np.int32))
    else:
        TRIU_INDICES[k] = np.triu_indices(k)

def construct():
    start_time = time.time()
    DEADLINE = 55.0
    
    max_val = 4096
    max_sum = 3 * max_val + 1
    
    # Deterministic Greedy
    used_sums = np.zeros(max_sum, dtype=bool)
    selected = []
    S = np.array([], dtype=np.int32)
    
    for x in range(1, max_val + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        k = len(S)
        if k == 0:
            selected.append(x)
            S = np.array([x], dtype=np.int32)
            used_sums[3*x] = True
            continue
            
        i_idx, j_idx = TRIU_INDICES[k]
        num_pairs = len(i_idx)
        total_new = 1 + k + num_pairs
        
        new_sums = np.empty(total_new, dtype=np.int32)
        new_sums[0] = 3 * x
        new_sums[1:1+k] = 2 * x + S
        if k > 1:
            new_sums[1+k:] = x + S[i_idx] + S[j_idx]
            
        if np.any(used_sums[new_sums]):
            continue
        if len(np.unique(new_sums)) != len(new_sums):
            continue
            
        selected.append(x)
        S = np.append(S, x)
        used_sums[new_sums] = True

    best_set = selected[:]
    best_size = len(best_set)
    
    # Randomized Restarts
    candidates = list(range(1, max_val + 1))
    
    while time.time() - start_time < DEADLINE - 1.0:
        # Shuffle
        random.shuffle(candidates)
        
        cur_set = []
        cur_S = np.array([], dtype=np.int32)
        cur_used = np.zeros(max_sum, dtype=bool)
        
        for x in candidates:
            k = len(cur_S)
            
            if k == 0:
                cur_set.append(x)
                cur_S = np.array([x], dtype=np.int32)
                cur_used[3*x] = True
                continue
            
            i_idx, j_idx = TRIU_INDICES[k]
            num_pairs = len(i_idx)
            total_new = 1 + k + num_pairs
            
            new_sums = np.empty(total_new, dtype=np.int32)
            new_sums[0] = 3 * x
            new_sums[1:1+k] = 2 * x + cur_S
            if k > 1:
                new_sums[1+k:] = x + cur_S[i_idx] + cur_S[j_idx]
                
            if np.any(cur_used[new_sums]):
                continue
            if len(np.unique(new_sums)) != len(new_sums):
                continue
                
            cur_set.append(x)
            cur_S = np.append(cur_S, x)
            cur_used[new_sums] = True
            
        if len(cur_set) > best_size:
            best_set = cur_set
            best_size = len(best_set)
            
            if best_size >= 40:
                break

    return best_set
