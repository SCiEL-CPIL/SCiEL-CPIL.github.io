import itertools
import random
import time
import sys
import numpy as np

# Global cache
_CACHE = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    points = range(v)
    # Generate all blocks
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Generate all triples
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {tri: i for i, tri in enumerate(triples)}
    n_triples = len(triples)
    
    # Precompute masks for each block (bit i is 1 if block covers triple i)
    block_masks = [0] * n_blocks
    # Also store list of triple indices for each block for faster count updates
    block_triple_indices = [None] * n_blocks
    
    for i, block in enumerate(all_blocks):
        mask = 0
        indices = []
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_triple_indices[i] = indices
        
    # Precompute for each triple, which blocks cover it
    # This is useful for finding replacement blocks in local search
    triple_to_blocks = [[] for _ in range(n_triples)]
    for i in range(n_blocks):
        for idx in block_triple_indices[i]:
            triple_to_blocks[idx].append(i)
            
    full_mask = (1 << n_triples) - 1
    
    _CACHE[key] = (all_blocks, block_masks, block_triple_indices, triple_to_blocks, full_mask, n_triples)
    return _CACHE[key]

def _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, seed):
    rng = random.Random(seed)
    uncovered = full_mask
    blocks_used = []
    
    # Optimization: Pre-sort blocks or use a priority queue?
    # Simple greedy: pick max gain.
    # To add randomness, we can shuffle the list of blocks initially, 
    # or pick randomly among those with max gain.
    
    # For speed, we'll just iterate.
    # We can speed this up by maintaining a list of "active" blocks?
    # No, 346k blocks is too many to iterate fully every step if we do many steps.
    # But steps are ~70. 70 * 346k = 24M ops. In Python, this might be slow (1-2s).
    # We need a faster greedy.
    
    # Alternative: 
    # 1. Pick a random triple that is uncovered.
    # 2. Pick a random block covering it.
    # This is "randomized greedy" or "first-fit". It's faster but yields worse initial solutions.
    # However, local search can fix it.
    
    # Let's try a hybrid:
    # Calculate gain for all blocks is O(N_blocks * N_triples/word_size).
    # With bitmasks, gain = (mask & uncovered).bit_count().
    # Iterating 346k blocks and doing bit_count is fast in C (Python int bit_count is C).
    # 346k * 70 steps = 24M bit_counts. This should take < 1 second.
    
    while uncovered:
        best_block_idx = -1
        max_gain = 0
        
        # Randomize tie-breaking
        # We can collect top candidates? Or just pick first max and break?
        # To diversify, let's shuffle the indices we check? No, too slow.
        # Let's just pick the first one with max gain. 
        # To make it non-deterministic, we can add a small random noise to the gain?
        # gain + rng.random() * epsilon?
        # Or, we can iterate in a random order?
        # Shuffling 346k indices is expensive.
        
        # Strategy: 
        # 1. Find max_gain.
        # 2. Collect all indices with max_gain.
        # 3. Pick one randomly.
        
        candidates = []
        for idx in range(len(all_blocks)):
            gain = (block_masks[idx] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                candidates = [idx]
            elif gain == max_gain and max_gain > 0:
                candidates.append(idx)
                
        if not candidates:
            break # Should not happen
            
        best_block_idx = rng.choice(candidates)
        blocks_used.append(best_block_idx)
        uncovered &= ~block_masks[best_block_idx]
        
    return blocks_used

def _local_search(v, k, t, initial_blocks, block_masks, block_triple_indices, triple_to_blocks, full_mask, n_triples, deadline):
    # State: list of block indices
    current_blocks = list(initial_blocks)
    
    # Precompute coverage counts
    triple_counts = [0] * n_triples
    for b in current_blocks:
        for idx in block_triple_indices[b]:
            triple_counts[idx] += 1
            
    num_uncovered = sum(1 for c in triple_counts if c == 0)
    
    # Verify initial validity
    if num_uncovered != 0:
        return current_blocks # Return invalid, caller should handle or this shouldn't happen from greedy
        
    best_blocks = list(current_blocks)
    best_len = len(current_blocks)
    
    rng = random.Random(42)
    
    # Simulated Annealing parameters
    # We want to minimize length.
    # Secondary objective: minimize total coverage (sum of triple_counts) to make deletions easier?
    # Or maximize "uniqueness"?
    
    # Let's use a simple hill climbing with random restarts/swaps.
    # Move types:
    # 1. Delete a redundant block.
    # 2. Swap a block with another to maintain validity (cost 0 change in length).
    
    while time.time() < deadline:
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = list(current_blocks)
            
        # Try to delete
        # Pick a random block index
        if current_blocks:
            pos = rng.randint(0, len(current_blocks) - 1)
            b_out = current_blocks[pos]
            
            # Check if b_out is redundant
            # A block is redundant if for every triple it covers, there is another block in S covering it.
            # i.e., min(triple_counts[idx] for idx in block_triple_indices[b_out]) > 1
            redundant = True
            for idx in block_triple_indices[b_out]:
                if triple_counts[idx] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove it
                del current_blocks[pos]
                for idx in block_triple_indices[b_out]:
                    triple_counts[idx] -= 1
                # num_uncovered stays 0
                continue # Immediately try again
            
        # If no deletion possible, try a swap
        # Pick a random block to remove
        if not current_blocks:
            break
            
        pos = rng.randint(0, len(current_blocks) - 1)
        b_out = current_blocks[pos]
        
        # Determine which triples become uncovered if we remove b_out
        # These are triples where triple_counts[idx] == 1
        uncovered_triples = []
        for idx in block_triple_indices[b_out]:
            if triple_counts[idx] == 1:
                uncovered_triples.append(idx)
                
        if not uncovered_triples:
            # This block was redundant, but we missed it? Or we just checked?
            # If we just checked and it wasn't redundant, this list is not empty.
            # If it is empty, it means removing it doesn't uncover anything? 
            # That implies it was redundant.
            # We should have caught it above.
            # Let's assume we proceed to find a replacement.
            pass
            
        # We need a block b_in that covers all triples in uncovered_triples
        # And b_in is not in current_blocks (or is, but then it's a no-op swap? No, we removed b_out)
        
        # Find candidate blocks
        # Intersection of triple_to_blocks for each triple in uncovered_triples
        if uncovered_triples:
            # Start with the triple with fewest covering blocks
            min_triple = min(uncovered_triples, key=lambda x: len(triple_to_blocks[x]))
            candidates = set(triple_to_blocks[min_triple])
            
            for tri in uncovered_triples:
                if tri == min_triple:
                    continue
                candidates.intersection_update(triple_to_blocks[tri])
                if not candidates:
                    break
            
            # Filter out blocks already in current_blocks
            current_set = set(current_blocks)
            candidates.difference_update(current_set)
            
            if candidates:
                b_in = rng.choice(list(candidates))
                
                # Perform swap
                # Remove b_out
                del current_blocks[pos]
                for idx in block_triple_indices[b_out]:
                    triple_counts[idx] -= 1
                    
                # Add b_in
                current_blocks.append(b_in)
                for idx in block_triple_indices[b_in]:
                    triple_counts[idx] += 1
                # num_uncovered should remain 0 if b_in covers all uncovered_triples
                # We don't strictly need to update num_uncovered if we know it's 0 and swap preserves validity
                # But for safety, we could check.
                
        else:
            # No uncovered triples? This means b_out was redundant.
            # We should have deleted it.
            # Force delete
            del current_blocks[pos]
            for idx in block_triple_indices[b_out]:
                triple_counts[idx] -= 1

    return best_blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    all_blocks, block_masks, block_triple_indices, triple_to_blocks, full_mask, n_triples = _precompute(v, k, t)
    
    best_blocks_global = None
    best_len_global = float('inf')
    
    seed = 0
    # Run multiple greedy + local search instances
    while time.time() < deadline - 5: # Leave 5s for final check
        # Greedy
        greedy_blocks = _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, seed)
        seed += 1
        
        if not greedy_blocks:
            continue
            
        # Local Search
        # Give local search a time slice, e.g., 10s or until it finds a very good solution?
        # Better: run local search until a short time slice ends, then move to next seed.
        # Or run local search on the best found so far?
        # Let's give each run a fixed time slice, e.g., 10 seconds.
        ls_deadline = time.time() + 10
        
        ls_blocks = _local_search(
            v, k, t, 
            greedy_blocks, 
            block_masks, 
            block_triple_indices, 
            triple_to_blocks, 
            full_mask, 
            n_triples, 
            ls_deadline
        )
        
        if len(ls_blocks) < best_len_global:
            best_len_global = len(ls_blocks)
            best_blocks_global = ls_blocks
            
        # If we reach a very low number, we might want to spend more time on it?
        # For now, just keep trying.
        
    if best_blocks_global is None:
        # Fallback
        greedy_blocks = _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, 999)
        best_blocks_global = greedy_blocks
        
    result = [list(all_blocks[idx]) for idx in best_blocks_global]
    return result
