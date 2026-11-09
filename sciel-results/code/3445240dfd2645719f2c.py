import itertools
import random
import time
import sys

# Global cache for precomputed data, keyed by (v, k, t) to be safe if called multiple times with different params
# Though the task implies fixed params, it's good practice.
_CACHE = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    start_time = time.time()
    # Generate all blocks
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Map each triple to a unique bit index
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {tri: i for i, tri in enumerate(triples)}
    n_triples = len(triples)
    
    # Precompute masks for each block
    # A block mask is an integer where bit j is 1 if the block contains the j-th triple
    block_masks = [0] * n_blocks
    for i, block in enumerate(all_blocks):
        mask = 0
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    full_mask = (1 << n_triples) - 1
    
    _CACHE[key] = (all_blocks, block_masks, full_mask)
    return _CACHE[key]

def _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, seed):
    rng = random.Random(seed)
    uncovered = full_mask
    blocks_used = []
    used_mask = 0
    
    while uncovered:
        best_block_idx = -1
        max_gain = 0
        
        # To add some randomness and escape local optima, we can shuffle or pick randomly among top candidates
        # But standard greedy with tie-breaking by random order or just first found is a start.
        # Let's iterate all blocks and find the one with max gain.
        # To break ties randomly, we can collect top candidates.
        
        # Optimization: If we just pick the first max, we might be deterministic.
        # Let's add a small random noise or shuffle the order of evaluation?
        # Shuffling 24515 items every step is expensive.
        # Instead, let's just pick the first one with max_gain. 
        # To diversify, we can run multiple seeds.
        
        for idx in range(len(all_blocks)):
            gain = (block_masks[idx] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_block_idx = idx
                if gain == uncovered.bit_count():
                    # Perfect cover for remaining? Unlikely for large uncovered, but possible if uncovered is small.
                    # Actually, if gain == uncovered.bit_count(), we covered everything.
                    # We can break early? No, we still need to select this block.
                    # But we don't need to check others.
                    pass
            
        if best_block_idx == -1:
            # Should not happen if covering is possible
            break
            
        blocks_used.append(best_block_idx)
        used_mask |= block_masks[best_block_idx]
        uncovered &= ~used_mask
        
    return blocks_used

def _reduce_blocks(v, k, t, block_indices, block_masks, full_mask):
    if not block_indices:
        return []
        
    current_blocks = list(block_indices)
    changed = True
    
    while changed:
        changed = False
        new_current = []
        n = len(current_blocks)
        
        # Pre-compute union of all? 
        # If we remove block i, the union of others is (total_union ^ block_masks[i]) only if block i is unique?
        # No, XOR doesn't work for union if bits overlap.
        # We need OR of all others.
        # Since N is small (~76), O(N^2) is fine.
        
        for i, idx in enumerate(current_blocks):
            # Check if block i is redundant
            # Union of all blocks except i
            union_others = 0
            for j, other_idx in enumerate(current_blocks):
                if i != j:
                    union_others |= block_masks[other_idx]
            
            if (union_others & full_mask) == full_mask:
                # Redundant
                changed = True
            else:
                new_current.append(idx)
                
        current_blocks = new_current
        
    return current_blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # Precompute
    all_blocks, block_masks, full_mask = _precompute(v, k, t)
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    # Run greedy multiple times
    while time.time() < deadline - 10: # Leave 10s for reduction and final checks
        blocks = _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, seed)
        seed += 1
        
        if not blocks:
            continue
            
        # Reduce
        reduced_blocks = _reduce_blocks(v, k, t, blocks, block_masks, full_mask)
        
        if len(reduced_blocks) < best_len:
            best_len = len(reduced_blocks)
            best_blocks = reduced_blocks
            
            # If we hit a very good solution, we might want to try harder to reduce or just stop?
            # For now, just keep searching.
            
    # If we have a result, convert indices to actual block lists
    if best_blocks is None:
        # Fallback: return a simple valid covering if we failed (shouldn't happen)
        # Or just return the last non-reduced one?
        # Actually, if best_blocks is None, we need a valid covering.
        # Let's do one fast greedy and return it.
        blocks = _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, 999)
        best_blocks = blocks
        
    # Format output: list of lists of ints
    result = [list(all_blocks[idx]) for idx in best_blocks]
    
    return result
