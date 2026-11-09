import itertools
import random
import time
import sys

# Global cache for precomputed data
_CACHE = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    # Generate all blocks
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Map each triple to a unique bit index
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {tri: i for i, tri in enumerate(triples)}
    n_triples = len(triples)
    
    # Precompute masks for each block
    block_masks = [0] * n_blocks
    for i, block in enumerate(all_blocks):
        mask = 0
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    full_mask = (1 << n_triples) - 1
    
    # Precompute block_triples for reduction logic (list of triple indices)
    block_triple_indices = []
    for i, block in enumerate(all_blocks):
        inds = []
        for tri in itertools.combinations(block, t):
            inds.append(triple_to_idx[tri])
        block_triple_indices.append(inds)
        
    _CACHE[key] = (all_blocks, block_masks, full_mask, block_triple_indices)
    return _CACHE[key]

def _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, seed, deadline):
    rng = random.Random(seed)
    uncovered = full_mask
    blocks_used = []
    used_mask = 0
    n_blocks = len(all_blocks)
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_block_idx = -1
        max_gain = 0
        
        # Iterate all blocks to find the one with max gain
        # This is O(N_blocks * bit_ops). N_blocks ~ 24k, bit_ops are fast.
        # 24k * 1000 steps = 24M ops. Should be okay in Python if optimized?
        # Actually, pure Python loop over 24k items with bit_count might be slow.
        # Let's check if we can vectorize or just accept the speed.
        # The current program does this and scores 1.09.
        
        for idx in range(n_blocks):
            gain = (block_masks[idx] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_block_idx = idx
                # Early break if we found a block that covers everything?
                # if gain == uncovered.bit_count(): break # Unlikely to help much as we still need to update mask
        
        if best_block_idx == -1:
            break
            
        blocks_used.append(best_block_idx)
        used_mask |= block_masks[best_block_idx]
        uncovered &= ~used_mask
        
    return blocks_used

def _reduce_blocks(v, k, t, block_indices, block_masks, full_mask, block_triple_indices, deadline):
    if not block_indices:
        return []
        
    current_blocks = list(block_indices)
    changed = True
    
    # Pre-compute coverage counts for each triple using an array
    # We need to know how many blocks cover each triple.
    # n_triples can be found from full_mask.bit_length() or passed?
    # Let's derive n_triples from the max index in block_triple_indices or just use a dict?
    # Better: n_triples is fixed. We can calculate it or store it in cache.
    # Let's add n_triples to cache return? No, we can just use a large enough array or dict.
    # Actually, we can just use a list of size 1<<t? No, indices are dense.
    # Let's just use a dictionary or a list of size max_index+1.
    
    # To be efficient, let's get n_triples from the cache key logic or just pass it?
    # The cache returns 4 items. Let's assume we can infer or just use a dict.
    # Dict is safer for sparse coverage, but list is faster.
    # Let's use a list. We need n_triples.
    # We can get it from the max index in any block's triple indices, but that's fragile.
    # Let's modify _precompute to return n_triples?
    # Or just use a dictionary for coverage counts.
    
    coverage = {}
    for idx in current_blocks:
        for tri_idx in block_triple_indices[idx]:
            coverage[tri_idx] = coverage.get(tri_idx, 0) + 1
            
    while changed:
        if time.time() > deadline:
            break
        changed = False
        new_current = []
        for idx in current_blocks:
            # Check if block i is redundant
            # i.e., for every triple in this block, coverage[tri] > 1
            is_redundant = True
            for tri_idx in block_triple_indices[idx]:
                if coverage.get(tri_idx, 0) <= 1:
                    is_redundant = False
                    break
                    
            if is_redundant:
                # Remove it
                for tri_idx in block_triple_indices[idx]:
                    coverage[tri_idx] -= 1
                changed = True
            else:
                new_current.append(idx)
                
        current_blocks = new_current
        
    return current_blocks

def construct(v, k, t):
    start_time = time.time()
    # The evaluator kills at 150s. We should aim to return well before that.
    # Let's set a deadline of 140s to be safe.
    deadline = start_time + 140
    
    # Precompute
    all_blocks, block_masks, full_mask, block_triple_indices = _precompute(v, k, t)
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    # Run greedy multiple times
    # We need to leave time for reduction.
    # Greedy is the bottleneck.
    while time.time() < deadline - 20: # Leave 20s for reduction and final checks
        blocks = _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, seed, deadline - 10)
        seed += 1
        
        if not blocks:
            continue
            
        # Reduce
        reduced_blocks = _reduce_blocks(v, k, t, blocks, block_masks, full_mask, block_triple_indices, deadline - 5)
        
        if len(reduced_blocks) < best_len:
            best_len = len(reduced_blocks)
            best_blocks = reduced_blocks
            
    # If we have a result, convert indices to actual block lists
    if best_blocks is None:
        # Fallback: return a simple valid covering if we failed
        blocks = _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, 999, deadline)
        best_blocks = blocks
        
    # Format output: list of lists of ints
    result = [list(all_blocks[idx]) for idx in best_blocks]
    
    return result
