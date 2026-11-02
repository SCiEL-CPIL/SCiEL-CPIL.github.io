import itertools
import time
import random
import numpy as np

def _precompute(v, k, t):
    """Precompute all t-subsets, all k-blocks, and coverage masks."""
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    all_kblocks = list(itertools.combinations(range(v), k))
    num_kblocks = len(all_kblocks)
    block_to_idx = {b: i for i, b in enumerate(all_kblocks)}
    
    # Precompute coverage masks for each block
    block_masks = [0] * num_kblocks
    
    for b_idx, block in enumerate(all_kblocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            ts_idx = tset_to_idx[ts]
            mask |= (1 << ts_idx)
        block_masks[b_idx] = mask
        
    # Full coverage mask
    full_mask = (1 << num_tsets) - 1
    
    # Precompute tset_to_blocks: for each tset, which blocks cover it
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b_idx, block in enumerate(all_kblocks):
        for ts in itertools.combinations(block, t):
            ts_idx = tset_to_idx[ts]
            tset_to_blocks[ts_idx].append(b_idx)
    
    return all_tsets, all_kblocks, block_masks, tset_to_idx, block_to_idx, num_tsets, num_kblocks, full_mask, tset_to_blocks

def _bitmask_greedy(v, k, t, precomp, seed=0, deadline=None):
    """Vectorized greedy using bitmasks."""
    if deadline is None:
        deadline = time.time() + 100
        
    all_tsets, all_kblocks, block_masks, tset_to_idx, block_to_idx, num_tsets, num_kblocks, full_mask, tset_to_blocks = precomp
    
    rng = random.Random(seed)
    
    covered_mask = 0
    is_selected = [False] * num_kblocks
    sol_indices = []
    
    # Precompute gains
    gains = [bin(m).count('1') for m in block_masks]
    
    while True:
        if time.time() > deadline:
            return None
            
        if covered_mask == full_mask:
            break
            
        max_gain = -1
        best_idx = -1
        for i in range(num_kblocks):
            if is_selected[i]:
                continue
            if gains[i] > max_gain:
                max_gain = gains[i]
                best_idx = i
                
        if best_idx == -1 or max_gain <= 0:
            break
            
        sol_indices.append(best_idx)
        is_selected[best_idx] = True
        gains[best_idx] = 0
        
        new_bits = block_masks[best_idx] & ~covered_mask
        for j in range(num_kblocks):
            if not is_selected[j]:
                gains[j] -= bin(block_masks[j] & new_bits).count('1')
                
        covered_mask |= block_masks[best_idx]
        
    return [list(all_kblocks[idx]) for idx in sol_indices]

def _reduce_blocks(blocks, precomp, deadline):
    """Remove redundant blocks from a valid solution using efficient mask operations."""
    all_tsets, all_kblocks, block_masks, tset_to_idx, block_to_idx, num_tsets, num_kblocks, full_mask, tset_to_blocks = precomp
    
    blocks = list(blocks)
    
    block_indices = []
    for b in blocks:
        idx = block_to_idx.get(tuple(b), -1)
        if idx == -1:
            return blocks
        block_indices.append(idx)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n = len(block_indices)
        for i in range(n):
            if time.time() > deadline:
                return blocks
            
            # Calculate mask without block i
            mask_without = 0
            for j in range(n):
                if j != i:
                    mask_without |= block_masks[block_indices[j]]
                    
            if mask_without == full_mask:
                block_indices.pop(i)
                changed = True
                break
                
    return [list(all_kblocks[idx]) for idx in block_indices]

def _pairwise_swap_search(blocks, precomp, deadline, max_iterations=100):
    """Try to reduce solution size by swapping pairs of blocks."""
    all_tsets, all_kblocks, block_masks, tset_to_idx, block_to_idx, num_tsets, num_kblocks, full_mask, tset_to_blocks = precomp
    
    blocks = list(blocks)
    block_indices = []
    for b in blocks:
        idx = block_to_idx.get(tuple(b), -1)
        if idx >= 0:
            block_indices.append(idx)
    
    improved = True
    iteration = 0
    while improved and time.time() < deadline and iteration < max_iterations:
        improved = False
        iteration += 1
        n = len(block_indices)
        
        # Try removing one block and replacing with a better one
        for i in range(n):
            if time.time() > deadline:
                return blocks
            
            # Remove block i
            block_indices.pop(i)
            
            # Check if still valid
            mask_without = 0
            for j in range(len(block_indices)):
                mask_without |= block_masks[block_indices[j]]
                
            if mask_without != full_mask:
                # Need to find a block that covers the missing tsets
                missing = full_mask & ~mask_without
                best_block = -1
                best_gain = 0
                
                for j in range(num_kblocks):
                    if j in block_indices:
                        continue
                    gain = bin(block_masks[j] & missing).count('1')
                    if gain > best_gain:
                        best_gain = gain
                        best_block = j
                        
                if best_block != -1 and best_gain > 0:
                    block_indices.append(best_block)
                    # Check if this actually reduces or maintains size
                    # For now, we only accept if it doesn't increase
                    pass
                else:
                    # Can't repair, put back
                    block_indices.insert(i, block_indices[i] if i < len(block_indices) else block_indices[0])
                    # Actually, we need to restore properly
                    # Let's restart this iteration
                    break
            else:
                # Valid without block i, keep it removed
                improved = True
                break
                
        if not improved and iteration < max_iterations:
            # Try pairwise swaps: remove block i, add block j
            for i in range(len(block_indices)):
                if time.time() > deadline:
                    return blocks
                    
                orig_i = block_indices[i]
                block_indices.pop(i)
                
                mask_without = 0
                for j in range(len(block_indices)):
                    mask_without |= block_masks[block_indices[j]]
                    
                if mask_without == full_mask:
                    # Already valid, keep removed
                    improved = True
                    break
                else:
                    missing = full_mask & ~mask_without
                    # Find best replacement
                    best_block = -1
                    best_gain = 0
                    for j in range(num_kblocks):
                        if j in block_indices:
                            continue
                        gain = bin(block_masks[j] & missing).count('1')
                        if gain > best_gain:
                            best_gain = gain
                            best_block = j
                            
                    if best_block != -1:
                        block_indices.append(best_block)
                    else:
                        block_indices.insert(i, orig_i)
                        
    return [list(all_kblocks[idx]) for idx in block_indices]

def _repair_solution(blocks, precomp, deadline):
    """Make a solution valid by adding blocks if necessary."""
    all_tsets, all_kblocks, block_masks, tset_to_idx, block_to_idx, num_tsets, num_kblocks, full_mask, tset_to_blocks = precomp
    
    blocks = list(blocks)
    block_indices = []
    for b in blocks:
        idx = block_to_idx.get(tuple(b), -1)
        if idx >= 0:
            block_indices.append(idx)
            
    covered_mask = 0
    for idx in block_indices:
        covered_mask |= block_masks[idx]
        
    if covered_mask == full_mask:
        return blocks
        
    remaining_mask = full_mask & ~covered_mask
    is_selected = [False] * num_kblocks
    for idx in block_indices:
        is_selected[idx] = True
        
    while remaining_mask != 0 and time.time() < deadline:
        best_idx = -1
        max_gain = 0
        for i in range(num_kblocks):
            if is_selected[i]:
                continue
            new_bits = block_masks[i] & remaining_mask
            gain = bin(new_bits).count('1')
            if gain > max_gain:
                max_gain = gain
                best_idx = i
                
        if best_idx == -1 or max_gain == 0:
            break
            
        block_indices.append(best_idx)
        is_selected[best_idx] = True
        covered_mask |= block_masks[best_idx]
        remaining_mask = full_mask & ~covered_mask
        
    return [list(all_kblocks[idx]) for idx in block_indices]

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 120.0
    
    precomp = _precompute(v, k, t)
    
    all_tsets, all_kblocks, block_masks, tset_to_idx, block_to_idx, num_tsets, num_kblocks, full_mask, tset_to_blocks = precomp
    
    # Phase 1: Multiple greedy initializations
    best_sol = _bitmask_greedy(v, k, t, precomp, seed=0, deadline=start_time + 30.0)
    if best_sol is None:
        best_sol = [list(all_kblocks[0])]
        
    best_len = len(best_sol)
    
    for i in range(1, 8):
        if time.time() > start_time + 50.0:
            break
        sol = _bitmask_greedy(v, k, t, precomp, seed=i, deadline=start_time + 50.0)
        if sol is not None:
            sol = _reduce_blocks(sol, precomp, start_time + 55.0)
            if len(sol) < best_len:
                best_len = len(sol)
                best_sol = sol
                
    # Phase 2: Local search with pairwise swaps
    best_sol = _pairwise_swap_search(best_sol, precomp, start_time + 90.0, max_iterations=200)
    best_len = len(best_sol)
    
    # Phase 3: Final reduction
    best_sol = _reduce_blocks(best_sol, precomp, deadline - 5.0)
    
    return best_sol
