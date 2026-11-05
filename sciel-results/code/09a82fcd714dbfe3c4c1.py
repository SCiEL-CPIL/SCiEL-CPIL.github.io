import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # 1. Precomputation
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    all_subsets = list(itertools.combinations(points, t))
    N_subsets = len(all_subsets)
    
    # Map subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute for each block:
    # 1. List of subset indices it covers
    # 2. Bitmask of subset indices it covers (Python int)
    block_subsets = []
    block_bitmasks = []
    
    # For fast candidate lookup: for each subset, which blocks cover it?
    # This is a list of lists. blocks_covering_subset[s] = [b1, b2, ...]
    blocks_covering_subset = [[] for _ in range(N_subsets)]
    
    for i, b in enumerate(all_blocks):
        subs = [subset_to_idx[s] for s in itertools.combinations(b, t)]
        block_subsets.append(subs)
        
        mask = 0
        for s in subs:
            mask |= (1 << s)
            blocks_covering_subset[s].append(i)
        block_bitmasks.append(mask)
        
    # Build a sparse matrix representation for the incidence matrix (blocks x subsets)
    # Using CSR-like structure for fast row access and column updates
    row_offsets = np.zeros(N_blocks + 1, dtype=np.int32)
    for i, subs in enumerate(block_subsets):
        row_offsets[i + 1] = row_offsets[i] + len(subs)
    row_offsets = np.cumsum(row_offsets)
    
    col_indices = np.zeros(row_offsets[-1], dtype=np.int32)
    pos = 0
    for subs in block_subsets:
        col_indices[pos:pos+len(subs)] = subs
        pos += len(subs)
    
    # Inverse: for each subset, list of blocks covering it (already have blocks_covering_subset)
    # Convert to numpy arrays for faster access
    blocks_covering_subset_np = [np.array(lst, dtype=np.int32) for lst in blocks_covering_subset]
    
    def run_greedy(seed=0):
        rng = random.Random(seed)
        cover_count = np.zeros(N_subsets, dtype=np.int32)
        potential = np.zeros(N_blocks, dtype=np.int32)
        for i, subs in enumerate(block_subsets):
            potential[i] = len(subs)
        selected_blocks = []
        selected_set = set()
        
        while np.any(cover_count == 0):
            if time.time() > deadline - 5.0:
                return None, None
                
            max_pot = np.max(potential)
            if max_pot == 0:
                break
                
            if seed == 0:
                best_idx = np.argmax(potential)
            else:
                candidates = np.where(potential == max_pot)[0]
                best_idx = candidates[rng.randint(0, len(candidates)-1)]
                
            selected_blocks.append(best_idx)
            selected_set.add(best_idx)
            
            # Update cover_count and potential
            for s in block_subsets[best_idx]:
                old_count = cover_count[s]
                cover_count[s] += 1
                if old_count == 0:
                    for b in blocks_covering_subset_np[s]:
                        potential[b] -= 1
                        
        return selected_blocks, cover_count

    def local_search(selected_blocks, cover_count):
        selected_blocks = list(selected_blocks)
        cover_count = cover_count.copy()
        selected_set = set(selected_blocks)
        
        best_len = len(selected_blocks)
        best_blocks = list(selected_blocks)
        
        steps = 0
        max_steps = 100000
        no_improve_count = 0
        max_no_improve = 5000
        
        while steps < max_steps:
            steps += 1
            if time.time() > deadline - 2.0:
                break
            if not selected_blocks:
                break
                
            b_out_idx = random.randint(0, len(selected_blocks) - 1)
            b_out = selected_blocks[b_out_idx]
            
            critical_subsets = []
            for s in block_subsets[b_out]:
                if cover_count[s] == 1:
                    critical_subsets.append(s)
            
            if not critical_subsets:
                for s in block_subsets[b_out]:
                    cover_count[s] -= 1
                selected_blocks.pop(b_out_idx)
                selected_set.remove(b_out)
                
                if len(selected_blocks) < best_len:
                    best_len = len(selected_blocks)
                    best_blocks = list(selected_blocks)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                critical_mask = 0
                for s in critical_subsets:
                    critical_mask |= (1 << s)
                
                first_crit = critical_subsets[0]
                candidates = blocks_covering_subset_np[first_crit]
                
                found_swap = False
                for b_in in candidates:
                    if b_in in selected_set:
                        continue
                    
                    if (block_bitmasks[b_in] & critical_mask) == critical_mask:
                        for s in block_subsets[b_out]:
                            cover_count[s] -= 1
                        selected_blocks.pop(b_out_idx)
                        selected_set.remove(b_out)
                        
                        selected_blocks.append(b_in)
                        selected_set.add(b_in)
                        for s in block_subsets[b_in]:
                            cover_count[s] += 1
                        
                        found_swap = True
                        break
                
                if found_swap:
                    changed = True
                    while changed:
                        changed = False
                        for i in range(len(selected_blocks) - 1, -1, -1):
                            b_idx = selected_blocks[i]
                            is_redundant = True
                            for s in block_subsets[b_idx]:
                                if cover_count[s] == 1:
                                    is_redundant = False
                                    break
                            if is_redundant:
                                for s in block_subsets[b_idx]:
                                    cover_count[s] -= 1
                                selected_blocks.pop(i)
                                selected_set.remove(b_idx)
                                changed = True
                                if len(selected_blocks) < best_len:
                                    best_len = len(selected_blocks)
                                    best_blocks = list(selected_blocks)
                                    no_improve_count = 0
                            
                    if len(selected_blocks) >= best_len:
                        no_improve_count += 1
                    else:
                        no_improve_count = 0
                else:
                    no_improve_count += 1
                    
            if no_improve_count > max_no_improve:
                break
                
        return best_blocks, best_len

    # Main execution
    best_blocks_global = None
    best_len_global = float('inf')
    
    # 1. Deterministic greedy
    blocks, counts = run_greedy(0)
    if blocks:
        best_blocks_local, best_len_local = local_search(blocks, counts)
        if best_len_local < best_len_global:
            best_len_global = best_len_local
            best_blocks_global = best_blocks_local
            
    # 2. Randomized greedy restarts
    for seed in range(1, 20):
        if time.time() > deadline - 10.0:
            break
            
        blocks, counts = run_greedy(seed)
        if not blocks:
            continue
            
        best_blocks_local, best_len_local = local_search(blocks, counts)
        
        if best_len_local < best_len_global:
            best_len_global = best_len_local
            best_blocks_global = best_blocks_local

    # 3. Final polish
    if best_blocks_global and time.time() < deadline - 5.0:
        cover_count_final = np.zeros(N_subsets, dtype=np.int32)
        for b in best_blocks_global:
            for s in block_subsets[b]:
                cover_count_final[s] += 1
        
        best_blocks_final, best_len_final = local_search(list(best_blocks_global), cover_count_final)
        if best_len_final < best_len_global:
            best_len_global = best_len_final
            best_blocks_global = best_blocks_final

    if not best_blocks_global:
        blocks, _ = run_greedy(0)
        best_blocks_global = blocks

    return [list(all_blocks[i]) for i in best_blocks_global]
