import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    mask_to_idx = {}
    for i, s in enumerate(all_t_subsets):
        mask = 0
        for x in s:
            mask |= (1 << x)
        mask_to_idx[mask] = i
        
    comb_mask = {}
    for comb in itertools.combinations(range(v), t - 1):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        comb_mask[comb] = mask
        
    uncovered = np.ones(n_t_subsets, dtype=bool)
    blocks = []
    
    point_to_indices_np = []
    for x in range(v):
        indices = [i for i, s in enumerate(all_t_subsets) if x in s]
        point_to_indices_np.append(np.array(indices, dtype=np.int64))
        
    while np.any(uncovered):
        if time.time() > deadline:
            break
            
        block = []
        current_block_set = set()
        
        for step in range(k):
            best_gain = -1
            best_point = -1
            candidates = [x for x in range(v) if x not in current_block_set]
            
            if len(current_block_set) < t - 1:
                for x in candidates:
                    # Gain is effectively "potential"
                    gain = np.sum(uncovered[point_to_indices_np[x]])
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
            else:
                t_minus_1_subsets = list(itertools.combinations(sorted(current_block_set), t - 1))
                
                for x in candidates:
                    gain = 0
                    x_bit = 1 << x
                    for T in t_minus_1_subsets:
                        mask_S = comb_mask[T] | x_bit
                        if mask_S in mask_to_idx:
                            idx = mask_to_idx[mask_S]
                            if uncovered[idx]:
                                gain += 1
                                
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
            
            if best_point == -1:
                if candidates:
                    best_point = candidates[0]
                else:
                    break
                    
            block.append(best_point)
            current_block_set.add(best_point)
            
        block.sort()
        blocks.append(block)
        
        for s in itertools.combinations(block, t):
            mask = 0
            for x in s:
                mask |= (1 << x)
            idx = mask_to_idx[mask]
            uncovered[idx] = False
            
    return blocks
