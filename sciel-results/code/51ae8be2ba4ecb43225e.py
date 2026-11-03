import itertools
import time
import random
import sys
import heapq
from collections import defaultdict
import numpy as np

def _deadline_ok(deadline):
    return time.time() < deadline

def _prune_redundant(selected, cover_matrix, n_t, deadline):
    """Remove redundant blocks using vectorized coverage counts."""
    if not selected:
        return selected
    
    sel_arr = np.array(selected, dtype=np.int64)
    counts = cover_matrix[sel_arr].sum(axis=0)
    
    changed = True
    while changed and _deadline_ok(deadline):
        changed = False
        # Find blocks where all covered subsets have count > 1
        for i in range(len(sel_arr)):
            block_idx = sel_arr[i]
            covered = cover_matrix[block_idx]
            if (counts[covered] > 1).all():
                # Remove this block
                counts[covered] -= 1
                sel_arr = np.delete(sel_arr, i)
                changed = True
                if not _deadline_ok(deadline):
                    break
    
    return list(sel_arr)

def _local_search(selected, cover_matrix, n_t, sub_to_blocks, deadline, max_iterations=500):
    """Local search: try removing blocks and finding minimal repairs via swaps."""
    if not selected or len(selected) < 2:
        return selected
    
    sel_set = set(selected)
    sel_arr = np.array(selected, dtype=np.int64)
    counts = cover_matrix[sel_arr].sum(axis=0)
    
    best_size = len(sel_arr)
    best_sel = sel_arr.copy()
    
    for iteration in range(max_iterations):
        if not _deadline_ok(deadline):
            break
        
        if len(sel_arr) < 2:
            break
        
        # Pick a random block to try removing
        idx = random.randint(0, len(sel_arr) - 1)
        block_to_remove = sel_arr[idx]
        
        # Find subsets uniquely covered by this block
        covered = cover_matrix[block_to_remove]
        unique_covered = covered[counts[covered] <= 1]
        
        if len(unique_covered) == 0:
            # Redundant block - remove it
            counts[covered] -= 1
            sel_arr = np.delete(sel_arr, idx)
            sel_set.discard(block_to_remove)
            best_size = len(sel_arr)
            best_sel = sel_arr.copy()
            continue
        
        # Try to find a single block that covers all unique_covered
        # Find the most constrained unique subset
        min_count = float('inf')
        target_sub = -1
        for s in unique_covered:
            c = len(sub_to_blocks[s])
            if c < min_count:
                min_count = c
                target_sub = s
                if c == 1:
                    break
        
        if target_sub == -1:
            continue
        
        candidates = sub_to_blocks[target_sub]
        best_swap = -1
        best_swap_score = -1
        
        for b in candidates:
            if b in sel_set or b == block_to_remove:
                continue
            
            # Check if this block covers all unique_covered
            b_covered = cover_matrix[b]
            # Count how many unique_covered are in b_covered
            b_covered_set = set(b_covered)
            covers_all = all(s in b_covered_set for s in unique_covered)
            
            if not covers_all:
                continue
            
            # Score: prefer blocks that add less redundant coverage
            # (i.e., blocks whose covered subsets are already well-covered)
            redundant = np.sum(counts[b_covered] >= 1)
            # We want to minimize new unique coverage, so maximize redundant
            if redundant > best_swap_score:
                best_swap_score = redundant
                best_swap = b
        
        if best_swap != -1:
            # Perform the swap
            old_covered = cover_matrix[block_to_remove]
            new_covered = cover_matrix[best_swap]
            counts[old_covered] -= 1
            counts[new_covered] += 1
            sel_arr[idx] = best_swap
            sel_set.discard(block_to_remove)
            sel_set.add(best_swap)
    
    return list(best_sel)

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 5.0  # Safety margin
    
    # 1. Generate all t-subsets and assign indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Generate all k-blocks and their covered t-subsets as sparse matrix
    n_blocks = len(list(itertools.combinations(range(v), k)))
    blocks = []
    cover_lists = []
    
    for block in itertools.combinations(range(v), k):
        covers = []
        for sub in itertools.combinations(block, t):
            covers.append(sub_index[sub])
        blocks.append(block)
        cover_lists.append(covers)
        
    if not _deadline_ok(deadline):
        return []
    
    # Build cover_matrix as numpy array of lists (for fast indexing)
    # Use a list of numpy arrays for each block
    cover_arrays = [np.array(c, dtype=np.int64) for c in cover_lists]
    
    # Build sub_to_blocks
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, covers in enumerate(cover_lists):
        for j in covers:
            sub_to_blocks[j].append(i)
    
    if not _deadline_ok(deadline):
        return []
    
    # Precompute: for each block, a boolean array of which t-subsets it covers
    # This is expensive but enables fast gain computation
    # Instead, use the cover_arrays directly
    
    best_selected = None
    best_size = float('inf')
    
    # Adaptive number of restarts based on available time
    elapsed = time.time() - start
    remaining = deadline - time.time()
    # Estimate time per restart from first restart, start with conservative
    num_restarts = max(4, int(remaining / 12.0))
    num_restarts = min(num_restarts, 15)
    
    for restart in range(num_restarts):
        if not _deadline_ok(deadline):
            break
        
        # Randomized greedy set cover
        uncovered = set(range(n_t))
        selected = set()
        
        while uncovered:
            if not _deadline_ok(deadline):
                break
            
            # Find the uncovered subset with fewest candidate blocks
            min_count = float('inf')
            target_sub = -1
            for s in uncovered:
                c = len(sub_to_blocks[s])
                if c < min_count:
                    min_count = c
                    target_sub = s
                    if c == 1:
                        break
            
            if target_sub == -1:
                break
            
            candidates = sub_to_blocks[target_sub]
            
            # Calculate gains for all candidates
            gains = []
            for b in candidates:
                gain = 0
                for s in cover_lists[b]:
                    if s in uncovered:
                        gain += 1
                gains.append((gain, b))
            
            # Sort by gain descending
            gains.sort(reverse=True)
            
            # Pick block
            if restart == 0:
                best_block = gains[0][1]
            else:
                top_k = min(10, len(gains))
                r = random.random()
                cumulative = 0
                best_block = gains[0][1]
                for idx in range(top_k):
                    w = 1.0 / (idx + 1)
                    cumulative += w
                    if r < cumulative:
                        best_block = gains[idx][1]
                        break
                else:
                    best_block = gains[min(top_k, len(gains)) - 1][1]
            
            selected.add(best_block)
            
            for s in cover_lists[best_block]:
                uncovered.discard(s)
        
        if not _deadline_ok(deadline):
            break
        
        # Prune redundant blocks
        selected_list = _prune_redundant(list(selected), cover_arrays, n_t, deadline)
        
        # Local search
        selected_list = _local_search(selected_list, cover_arrays, n_t, sub_to_blocks, deadline, max_iterations=200)
        
        # Update best
        if len(selected_list) < best_size:
            best_size = len(selected_list)
            best_selected = list(selected_list)
        
        # If we have time, do more local search on the best solution
        if _deadline_ok(deadline) and best_selected is not None:
            remaining_time = deadline - time.time()
            if remaining_time > 5.0:
                best_selected = _local_search(best_selected, cover_arrays, n_t, sub_to_blocks, deadline, max_iterations=300)
                if len(best_selected) < best_size:
                    best_size = len(best_selected)
    
    # Final pruning and local search
    if best_selected is not None:
        best_selected = _prune_redundant(best_selected, cover_arrays, n_t, deadline)
        
        if _deadline_ok(deadline):
            remaining_time = deadline - time.time()
            if remaining_time > 3.0:
                best_selected = _local_search(best_selected, cover_arrays, n_t, sub_to_blocks, deadline, max_iterations=400)
                best_selected = _prune_redundant(best_selected, cover_arrays, n_t, deadline)
    
    # Convert to result
    result = [list(blocks[i]) for i in best_selected] if best_selected else []
    
    return result
