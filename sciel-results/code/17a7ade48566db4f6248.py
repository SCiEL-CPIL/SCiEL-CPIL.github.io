import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    # 1. Precomputation
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {ts: i for i, ts in enumerate(all_t_subs)}
    num_t_subs = len(all_t_subs)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    c_k_t = int(len(list(itertools.combinations(range(k), t))))
    block_t_indices = np.zeros((num_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, ts in enumerate(itertools.combinations(block, t)):
             block_t_indices[i, j] = t_sub_to_idx[ts]
             
    # 2. Greedy Initialization
    uncovered = np.ones(num_t_subs, dtype=np.int8)
    blocks_used = []
    used_block_mask = np.zeros(num_blocks, dtype=bool)
    
    while uncovered.sum() > 0:
        gains = uncovered[block_t_indices].sum(axis=1)
        gains[used_block_mask] = -1
        best_idx = np.argmax(gains)
        if gains[best_idx] <= 0:
            break
        blocks_used.append(best_idx)
        used_block_mask[best_idx] = True
        uncovered[block_t_indices[best_idx]] = 0
        
    current_blocks = np.array(blocks_used, dtype=np.int32)
    t_counts = np.zeros(num_t_subs, dtype=np.int32)
    for b_idx in current_blocks:
        t_counts[block_t_indices[b_idx]] += 1
        
    best_blocks = current_blocks.copy()
    best_size = len(current_blocks)
    
    # 3. Simulated Annealing
    LAMBDA = 100
    start_time = time.time()
    T = 10.0
    T_min = 0.1
    cooling = 0.99
    max_time = 110
    
    current_energy = len(current_blocks) + LAMBDA * np.sum(t_counts == 0)
    current_uncovered = np.sum(t_counts == 0)
    
    # Helper to update counts and energy
    def apply_remove(b_idx):
        nonlocal current_uncovered
        idxs = block_t_indices[b_idx]
        for idx in idxs:
            t_counts[idx] -= 1
            if t_counts[idx] == 0:
                current_uncovered += 1
        used_block_mask[b_idx] = False
        return len(current_blocks) - 1 + LAMBDA * current_uncovered

    def apply_add(b_idx):
        nonlocal current_uncovered
        idxs = block_t_indices[b_idx]
        for idx in idxs:
            if t_counts[idx] == 0:
                current_uncovered -= 1
            t_counts[idx] += 1
        used_block_mask[b_idx] = True
        return len(current_blocks) + 1 + LAMBDA * current_uncovered

    while T > T_min and (time.time() - start_time) < max_time:
        move_type = random.random()
        
        if len(current_blocks) == 0:
            b_idx = random.randint(0, num_blocks - 1)
            if used_block_mask[b_idx]:
                continue
            
            # Tentative Add
            old_uncovered = current_uncovered
            old_energy = current_energy
            old_len = len(current_blocks)
            
            # Apply
            idxs = block_t_indices[b_idx]
            for idx in idxs:
                if t_counts[idx] == 0:
                    current_uncovered -= 1
                t_counts[idx] += 1
            used_block_mask[b_idx] = True
            new_blocks = np.append(current_blocks, b_idx)
            current_blocks = new_blocks
            current_energy = len(current_blocks) + LAMBDA * current_uncovered
            
            if current_energy < old_energy or random.random() < np.exp((old_energy - current_energy) / T):
                if len(current_blocks) < best_size and current_uncovered == 0:
                    best_blocks = current_blocks.copy()
                    best_size = len(current_blocks)
            else:
                # Revert
                for idx in idxs:
                    t_counts[idx] -= 1
                    if t_counts[idx] == 0:
                        current_uncovered += 1
                used_block_mask[b_idx] = False
                current_blocks = current_blocks[:-1]
                current_energy = old_energy
                current_uncovered = old_uncovered
        
        else:
            if move_type < 0.4:
                # Remove
                pos = random.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[pos]
                
                old_uncovered = current_uncovered
                old_energy = current_energy
                
                # Apply Remove
                idxs = block_t_indices[b_idx]
                for idx in idxs:
                    t_counts[idx] -= 1
                    if t_counts[idx] == 0:
                        current_uncovered += 1
                used_block_mask[b_idx] = False
                new_blocks = np.delete(current_blocks, pos)
                current_blocks = new_blocks
                current_energy = len(current_blocks) + LAMBDA * current_uncovered
                
                if current_energy < old_energy or random.random() < np.exp((old_energy - current_energy) / T):
                    if len(current_blocks) < best_size and current_uncovered == 0:
                        best_blocks = current_blocks.copy()
                        best_size = len(current_blocks)
                else:
                    # Revert
                    for idx in idxs:
                        t_counts[idx] += 1
                        if t_counts[idx] == 1:
                            current_uncovered -= 1
                    used_block_mask[b_idx] = True
                    current_blocks = np.insert(current_blocks, pos, b_idx)
                    current_energy = old_energy
                    current_uncovered = old_uncovered
                    
            elif move_type < 0.7:
                # Add
                b_idx = random.randint(0, num_blocks - 1)
                if used_block_mask[b_idx]:
                    continue
                
                old_uncovered = current_uncovered
                old_energy = current_energy
                
                # Apply Add
                idxs = block_t_indices[b_idx]
                for idx in idxs:
                    if t_counts[idx] == 0:
                        current_uncovered -= 1
                    t_counts[idx] += 1
                used_block_mask[b_idx] = True
                new_blocks = np.append(current_blocks, b_idx)
                current_blocks = new_blocks
                current_energy = len(current_blocks) + LAMBDA * current_uncovered
                
                if current_energy < old_energy or random.random() < np.exp((old_energy - current_energy) / T):
                    if len(current_blocks) < best_size and current_uncovered == 0:
                        best_blocks = current_blocks.copy()
                        best_size = len(current_blocks)
                else:
                    # Revert
                    for idx in idxs:
                        t_counts[idx] -= 1
                        if t_counts[idx] == 0:
                            current_uncovered += 1
                    used_block_mask[b_idx] = False
                    current_blocks = current_blocks[:-1]
                    current_energy = old_energy
                    current_uncovered = old_uncovered
                    
            else:
                # Swap
                if len(current_blocks) < 2:
                    continue
                pos = random.randint(0, len(current_blocks) - 1)
                old_b_idx = current_blocks[pos]
                
                new_b_idx = random.randint(0, num_blocks - 1)
                if used_block_mask[new_b_idx]:
                    continue
                
                old_uncovered = current_uncovered
                old_energy = current_energy
                
                # Apply Swap: Remove old, Add new
                idxs_old = block_t_indices[old_b_idx]
                idxs_new = block_t_indices[new_b_idx]
                
                for idx in idxs_old:
                    t_counts[idx] -= 1
                    if t_counts[idx] == 0:
                        current_uncovered += 1
                used_block_mask[old_b_idx] = False
                
                for idx in idxs_new:
                    if t_counts[idx] == 0:
                        current_uncovered -= 1
                    t_counts[idx] += 1
                used_block_mask[new_b_idx] = True
                
                new_blocks = current_blocks.copy()
                new_blocks[pos] = new_b_idx
                current_blocks = new_blocks
                current_energy = len(current_blocks) + LAMBDA * current_uncovered
                
                if current_energy < old_energy or random.random() < np.exp((old_energy - current_energy) / T):
                    if len(current_blocks) < best_size and current_uncovered == 0:
                        best_blocks = current_blocks.copy()
                        best_size = len(current_blocks)
                else:
                    # Revert
                    for idx in idxs_old:
                        t_counts[idx] += 1
                        if t_counts[idx] == 1:
                            current_uncovered -= 1
                    used_block_mask[old_b_idx] = True
                    
                    for idx in idxs_new:
                        t_counts[idx] -= 1
                        if t_counts[idx] == 0:
                            current_uncovered += 1
                    used_block_mask[new_b_idx] = False
                    
                    current_blocks = current_blocks.copy()
                    current_blocks[pos] = old_b_idx
                    current_energy = old_energy
                    current_uncovered = old_uncovered
                    
        T *= cooling

    # 4. Local Search on Best (Deletion)
    final_blocks = best_blocks.copy()
    changed = True
    while changed:
        changed = False
        for i in range(len(final_blocks) - 1, -1, -1):
            b_idx = final_blocks[i]
            idxs = block_t_indices[b_idx]
            
            # Check if removal is safe
            safe = True
            for idx in idxs:
                if t_counts[idx] == 1:
                    safe = False
                    break
            
            if safe:
                for idx in idxs:
                    t_counts[idx] -= 1
                final_blocks = np.delete(final_blocks, i)
                changed = True
                break

    blocks = [list(all_blocks[b]) for b in final_blocks]
    return blocks
