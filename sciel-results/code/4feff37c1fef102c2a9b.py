import itertools
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute coverage lists for each block
    block_cover = []
    for B in all_kblocks:
        covered_indices = []
        for ts in itertools.combinations(B, t):
            covered_indices.append(tsub_to_idx[ts])
        block_cover.append(covered_indices)
        
    # Precompute map from t-subset index to list of block indices covering it
    tsub_to_blocks = defaultdict(list)
    for i, covs in enumerate(block_cover):
        for idx in covs:
            tsub_to_blocks[idx].append(i)
            
    best_blocks = None
    best_count = float('inf')
    
    # Use a few random restarts of a fast randomized greedy
    num_restarts = 3
    
    for _ in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
            
        # Fast Randomized Greedy
        selected_set = set()
        current_selected = []
        uncovered_list = list(range(n_tsubs))
        idx_to_pos = {i: i for i in range(n_tsubs)}
        
        while len(uncovered_list) > 0 and time.time() < deadline - 20.0:
            pos = np.random.randint(len(uncovered_list))
            t_idx = uncovered_list[pos]
            
            candidates = tsub_to_blocks[t_idx]
            if not candidates:
                break
                
            # Pick a random block covering t_idx
            b_idx = candidates[np.random.randint(len(candidates))]
            
            if b_idx in selected_set:
                # This shouldn't happen if logic is correct, but skip if it does
                continue
                
            selected_set.add(b_idx)
            current_selected.append(b_idx)
            
            # Remove covered t-subsets from uncovered_list
            for c_idx in block_cover[b_idx]:
                if c_idx in idx_to_pos:
                    c_pos = idx_to_pos[c_idx]
                    last_idx = uncovered_list[-1]
                    
                    # Swap c_idx with last
                    uncovered_list[c_pos] = last_idx
                    idx_to_pos[last_idx] = c_pos
                    
                    # Remove last
                    uncovered_list.pop()
                    del idx_to_pos[c_idx]
                    
        # Local Optimization
        if time.time() < deadline - 5.0 and current_selected:
            current_selected = _local_optimize_fast(current_selected, block_cover, tsub_to_blocks, n_tsubs, deadline - 2.0)
            
        if len(current_selected) < best_count:
            best_count = len(current_selected)
            best_blocks = current_selected
            
    if best_blocks is None:
        return []
        
    return [all_kblocks[i] for i in best_blocks]

def _local_optimize_fast(selected_indices, block_cover, tsub_to_blocks, n_tsubs, deadline):
    """
    Fast local optimization:
    1. Remove redundant blocks.
    2. Try 1-for-1 swaps to maintain coverage and potentially allow further removals.
    """
    if not selected_indices:
        return selected_indices
        
    # Compute cover counts
    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in selected_indices:
        for t_idx in block_cover[idx]:
            cover_counts[t_idx] += 1
            
    selected_set = set(selected_indices)
    
    # Phase 1: Remove redundant
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_selected = []
        for idx in selected_indices:
            essential = False
            for t_idx in block_cover[idx]:
                if cover_counts[t_idx] == 1:
                    essential = True
                    break
            if essential:
                new_selected.append(idx)
            else:
                for t_idx in block_cover[idx]:
                    cover_counts[t_idx] -= 1
                selected_set.remove(idx)
                changed = True
        selected_indices = new_selected
        
    # Phase 2: 1-for-1 Swaps
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i, idx_a in enumerate(selected_indices):
            if time.time() > deadline - 2.0:
                break
            
            unique_tsubs_a = [t_idx for t_idx in block_cover[idx_a] if cover_counts[t_idx] == 1]
            
            if not unique_tsubs_a:
                continue
                
            # Find a block B that covers all unique_tsubs_a
            cand_set = set(tsub_to_blocks[unique_tsubs_a[0]])
            for t_idx in unique_tsubs_a[1:]:
                cand_set.intersection_update(tsub_to_blocks[t_idx])
                
            cand_set -= selected_set
            
            if not cand_set:
                continue
                
            # Pick a candidate
            b_idx = list(cand_set)[0]
            
            # Perform swap
            for t_idx in block_cover[idx_a]:
                cover_counts[t_idx] -= 1
            selected_set.remove(idx_a)
            
            for t_idx in block_cover[b_idx]:
                cover_counts[t_idx] += 1
            selected_set.add(b_idx)
            
            selected_indices[i] = b_idx
            improved = True
            
    return selected_indices
