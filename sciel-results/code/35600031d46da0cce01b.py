import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140
    
    # Precomputation
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    triple_to_idx = {trip: i for i, trip in enumerate(triples)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute bitmask for each block
    # Bit i is set if triple i is in the block
    block_masks = [0] * num_blocks
    for idx, b in enumerate(all_blocks):
        mask = 0
        # Generate triples in b
        # b is sorted
        for i in range(len(b)):
            for j in range(i+1, len(b)):
                for l in range(j+1, len(b)):
                    trip = (b[i], b[j], b[l])
                    t_idx = triple_to_idx[trip]
                    mask |= (1 << t_idx)
        block_masks[idx] = mask
        
    # Initial full mask
    full_mask = (1 << num_triples) - 1
    
    def greedy_cover(uncovered_mask):
        """
        Greedily add blocks to cover all triples.
        Returns list of block indices added.
        """
        added_blocks = []
        curr_uncovered = uncovered_mask
        
        while curr_uncovered:
            if time.time() > deadline:
                break
            
            best_block = -1
            max_gain = -1
            # Find block with max gain
            # Optimization: if max_gain reaches k*t (max possible), we can break?
            # Max gain for a block is C(k,t) = 21.
            # If we find 21, it's perfect.
            for b in range(num_blocks):
                # Calculate gain
                # gain = number of bits in (curr_uncovered & block_masks[b])
                gain = (curr_uncovered & block_masks[b]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_block = b
                    if max_gain == 21: # C(7,3)
                        break
            
            if best_block == -1:
                break
                
            added_blocks.append(best_block)
            curr_uncovered &= ~block_masks[best_block]
            
        return added_blocks

    def prune_blocks(blocks):
        """
        Remove redundant blocks from the list.
        Returns pruned list.
        """
        # Calculate coverage count for each triple
        # Since we only have block indices, we need to rebuild coverage or use a different metric.
        # A block is redundant if removing it doesn't uncover any triple.
        # i.e., for every triple in block b, there is another block in the set that covers it.
        
        # Efficient check:
        # Let U be the union of triples covered by blocks in `blocks`.
        # If we remove b, the new union is U \ (Triples(b) \ (Union of others)).
        # Simpler: 
        # For each block b, check if Triples(b) is subset of Union(Others).
        
        # We can represent the union of all blocks as a bitmask `total_covered`.
        # But we need to know if a specific triple is covered by *another* block.
        
        # Let's use a count array or a "second best" approach.
        # Given the small number of blocks (~76), we can just do pairwise checks or simple reconstruction.
        
        # Reconstruct coverage counts is O(N_blocks * 21).
        # 76 * 21 = 1596 operations. Very fast.
        
        # Let's just rebuild counts for the current set.
        counts = [0] * num_triples
        for b in blocks:
            mask = block_masks[b]
            # Iterate set bits? Or just use the precomputed list of triple indices?
            # We didn't store triple indices list, only mask.
            # We can extract indices from mask?
            # Or we can store `block_triple_indices` as well.
            pass
        
        # Let's stick to the count array approach but we need triple indices.
        # I will generate `block_triple_indices` in precomputation.
        pass
        
    # Re-do precomputation to include indices for faster counting
    block_triple_indices = []
    for idx, b in enumerate(all_blocks):
        ts = []
        for i in range(len(b)):
            for j in range(i+1, len(b)):
                for l in range(j+1, len(b)):
                    trip = (b[i], b[j], b[l])
                    ts.append(triple_to_idx[trip])
        block_triple_indices.append(ts)

    def prune_blocks_opt(blocks):
        """
        Remove redundant blocks.
        """
        # Build counts
        counts = [0] * num_triples
        for b in blocks:
            for t_idx in block_triple_indices[b]:
                counts[t_idx] += 1
                
        changed = True
        while changed:
            changed = False
            for b in list(blocks):
                redundant = True
                for t_idx in block_triple_indices[b]:
                    if counts[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    blocks.remove(b)
                    for t_idx in block_triple_indices[b]:
                        counts[t_idx] -= 1
                    changed = True
        return blocks

    best_blocks = []
    best_size = float('inf')
    
    # Initial Greedy
    added = greedy_cover(full_mask)
    pruned = prune_blocks_opt(added)
    best_blocks = pruned
    best_size = len(pruned)
    
    # Local Search
    while time.time() < deadline:
        # Perturb
        curr_blocks = list(best_blocks)
        
        # Remove 1-2 random blocks
        if len(curr_blocks) > 0:
            num_removals = random.randint(1, 2)
            if len(curr_blocks) > num_removals:
                to_remove = random.sample(curr_blocks, num_removals)
                for b in to_remove:
                    curr_blocks.remove(b)
        
        # Calculate current uncovered mask
        # Start with full_mask and remove covered parts?
        # Easier: Start with 0 covered, add curr_blocks to find covered, then invert?
        # Or: Start with full_mask, and for each block in curr_blocks, clear its bits.
        curr_uncovered = full_mask
        for b in curr_blocks:
            curr_uncovered &= ~block_masks[b]
            
        # Fill
        added_more = greedy_cover(curr_uncovered)
        curr_blocks.extend(added_more)
        
        # Prune
        curr_blocks = prune_blocks_opt(curr_blocks)
        
        # Update best
        if len(curr_blocks) < best_size:
            best_size = len(curr_blocks)
            best_blocks = curr_blocks
            
    # Convert to output format
    result = []
    for b in best_blocks:
        result.append(list(all_blocks[b]))
        
    return result
