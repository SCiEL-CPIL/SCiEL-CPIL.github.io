import itertools
import random
import time
import math

_CACHE = {}

def _get_precomputed(v, k, t):
    if (v, k, t) in _CACHE:
        return _CACHE[(v, k, t)]
    
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    full_mask = (1 << num_t_subs) - 1
    
    k_blocks = list(itertools.combinations(range(v), k))
    block_masks = []
    for block in k_blocks:
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    _CACHE[(v, k, t)] = (k_blocks, block_masks, full_mask, t_sub_to_idx)
    return _CACHE[(v, k, t)]

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    k_blocks, block_masks, full_mask, t_sub_to_idx = _get_precomputed(v, k, t)
    num_blocks = len(k_blocks)
    
    # 1. Greedy Initialization
    uncovered = full_mask
    current_blocks = []
    
    while uncovered:
        best_gain = -1
        best_block_idx = -1
        for i in range(num_blocks):
            gain = bin(uncovered & block_masks[i]).count('1')
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
        
        if best_block_idx == -1:
            break
            
        current_blocks.append(best_block_idx)
        uncovered &= ~block_masks[best_block_idx]
        
        if time.time() > deadline - 10.0:
            break

    if not current_blocks:
        return []
        
    best_blocks = list(current_blocks)
    best_len = len(best_blocks)
    
    # 2. Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    def local_minimize(block_indices):
        changed = True
        while changed:
            changed = False
            for i in range(len(block_indices) - 1, -1, -1):
                idx_to_check = block_indices[i]
                mask_to_check = block_masks[idx_to_check]
                
                union_others = 0
                for j in range(len(block_indices)):
                    if j != i:
                        union_others |= block_masks[block_indices[j]]
                
                if (union_others | mask_to_check) == union_others:
                    block_indices.pop(i)
                    changed = True
                    break
        return block_indices

    current_blocks = list(best_blocks)
    
    while time.time() < deadline - 2.0:
        current_blocks = local_minimize(current_blocks)
        
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = list(current_blocks)
            
        move_type = random.random()
        
        if move_type < 0.4:
            if len(current_blocks) > 0:
                i = random.randint(0, len(current_blocks) - 1)
                old_idx = current_blocks[i]
                new_idx = random.randint(0, num_blocks - 1)
                
                if new_idx != old_idx:
                    union_others = 0
                    for j in range(len(current_blocks)):
                        if j != i:
                            union_others |= block_masks[current_blocks[j]]
                    
                    if (union_others | block_masks[new_idx]) == full_mask:
                        current_blocks[i] = new_idx
                        # Local minimize after swap
                        current_blocks = local_minimize(current_blocks)
                        
        elif move_type < 0.7:
            new_idx = random.randint(0, num_blocks - 1)
            if new_idx not in current_blocks:
                candidate_blocks = current_blocks + [new_idx]
                candidate_blocks = local_minimize(candidate_blocks)
                
                delta_len = len(candidate_blocks) - len(current_blocks)
                
                if delta_len < 0:
                    current_blocks = candidate_blocks
                else:
                    prob = math.exp(-delta_len / T) if T > 0 else 0
                    if random.random() < prob:
                        current_blocks = candidate_blocks
                        
        else:
            if len(current_blocks) > 0:
                i = random.randint(0, len(current_blocks) - 1)
                current_blocks.pop(i)
                
                union_current = 0
                for idx in current_blocks:
                    union_current |= block_masks[idx]
                
                uncovered = full_mask & ~union_current
                
                while uncovered:
                    best_gain = -1
                    best_block_idx = -1
                    for j in range(num_blocks):
                        gain = bin(uncovered & block_masks[j]).count('1')
                        if gain > best_gain:
                            best_gain = gain
                            best_block_idx = j
                    
                    if best_block_idx == -1 or best_gain == 0:
                        break
                        
                    current_blocks.append(best_block_idx)
                    uncovered &= ~block_masks[best_block_idx]
                
                current_blocks = local_minimize(current_blocks)
                
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = list(current_blocks)

        T *= cooling
        if T < min_T:
            T = min_T

    best_blocks = local_minimize(best_blocks)
    final_blocks = [list(k_blocks[idx]) for idx in best_blocks]
    
    return final_blocks
