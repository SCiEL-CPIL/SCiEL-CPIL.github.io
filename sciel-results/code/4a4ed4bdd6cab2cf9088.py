import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Precomputation
    # Generate all t-subsets (bitmasks)
    t_subsets = []
    tset_to_mask = {}
    for combo in itertools.combinations(range(v), t):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        t_subsets.append(mask)
        tset_to_mask[mask] = len(t_subsets) - 1
        
    n_tsets = len(t_subsets)
    
    # Generate all k-subsets (bitmasks) and their covered t-subsets
    block_list = []
    block_masks = []
    block_tsets = []  # list of lists of tset indices
    block_tsets_set = [] # list of sets of tset indices for fast lookup
    
    for combo in itertools.combinations(range(v), k):
        b_mask = 0
        for x in combo:
            b_mask |= (1 << x)
        
        # Find which t-subsets are in this block
        covered_tsets = []
        for tm in t_subsets:
            if (tm & b_mask) == tm:
                covered_tsets.append(tset_to_mask[tm])
        
        block_list.append(combo)
        block_masks.append(b_mask)
        block_tsets.append(covered_tsets)
        block_tsets_set.append(set(covered_tsets))
        
    n_blocks = len(block_list)
    
    # Inverse map: for each tset, list of blocks covering it
    tset_to_blocks = [[] for _ in range(n_tsets)]
    for b_idx in range(n_blocks):
        for t_idx in block_tsets[b_idx]:
            tset_to_blocks[t_idx].append(b_idx)
            
    # 2. Greedy Initialization
    uncovered = np.ones(n_tsets, dtype=bool)
    remaining = n_tsets
    selected = set()
    cov = np.zeros(n_tsets, dtype=np.int32)
    
    # Precompute gain for each block
    gain = np.zeros(n_blocks, dtype=np.int32)
    for b_idx in range(n_blocks):
        gain[b_idx] = len(block_tsets[b_idx])
        
    # Greedy loop
    while remaining > 0 and time.time() < deadline - 10.0:
        b_idx = int(np.argmax(gain))
        if gain[b_idx] <= 0:
            break
            
        selected.add(b_idx)
        
        # Update uncovered and cov
        for t_idx in block_tsets[b_idx]:
            if uncovered[t_idx]:
                uncovered[t_idx] = False
                remaining -= 1
            cov[t_idx] += 1
            
        # Update gains
        # To speed up, we can collect all blocks to update
        # But simple loop is likely fast enough for 6M ops
        for t_idx in block_tsets[b_idx]:
            for other_b in tset_to_blocks[t_idx]:
                if other_b != b_idx:
                    gain[other_b] -= 1
                    
        gain[b_idx] = -1
        
    # 3. Local Search
    def is_redundant(b_idx):
        for t_idx in block_tsets[b_idx]:
            if cov[t_idx] <= 1:
                return False
        return True
        
    def try_remove(b_idx):
        if is_redundant(b_idx):
            for t_idx in block_tsets[b_idx]:
                cov[t_idx] -= 1
            selected.remove(b_idx)
            return True
        return False
        
    def try_swap(b_idx):
        # Temporarily remove b_idx from cov
        deficit_tsets = []
        for t_idx in block_tsets[b_idx]:
            cov[t_idx] -= 1
            if cov[t_idx] == 0:
                deficit_tsets.append(t_idx)
                
        if not deficit_tsets:
            # Redundant
            selected.remove(b_idx)
            return True
            
        # Find replacement
        target_t = deficit_tsets[0]
        cand_tsets_set = block_tsets_set[b_idx] # Not needed for b_idx, but for candidates
        
        for cand_b in tset_to_blocks[target_t]:
            if cand_b in selected:
                continue
                
            # Check if cand_b covers all deficit_tsets
            covers_all = True
            cand_set = block_tsets_set[cand_b]
            for dt in deficit_tsets:
                if dt not in cand_set:
                    covers_all = False
                    break
                    
            if covers_all:
                # Perform swap
                selected.remove(b_idx)
                selected.add(cand_b)
                for t_idx in block_tsets[cand_b]:
                    cov[t_idx] += 1
                return True
                
        # Restore
        for t_idx in block_tsets[b_idx]:
            cov[t_idx] += 1
        return False
        
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        sel_list = list(selected)
        random.shuffle(sel_list)
        
        for b_idx in sel_list:
            if time.time() > deadline - 5.0:
                break
            if b_idx not in selected:
                continue
                
            if try_remove(b_idx):
                improved = True
                break
            if try_swap(b_idx):
                improved = True
                break
                
    final_blocks = [list(block_list[b_idx]) for b_idx in selected]
    return final_blocks
