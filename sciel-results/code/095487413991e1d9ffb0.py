"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using Greedy Set Cover with Bitsets
    and Iterative Redundancy Removal.
    """
    # 1. Generate all triples and map to indices
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # 2. Generate all blocks and their bitsets
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Pre-compute bitset for each block
    # A bitset is a Python int where bit i is 1 if triple i is covered by this block
    block_bitsets = []
    for block in all_blocks:
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            mask |= (1 << idx)
        block_bitsets.append(mask)
    
    # 3. Greedy Set Cover
    # We want to pick blocks to cover all triples (all bits set)
    full_mask = (1 << num_triples) - 1
    uncovered_mask = full_mask
    selected_blocks = []
    used_block_indices = set()
    
    # To speed up, we can maintain a list of candidate block indices
    # and filter them as we go. But simple iteration is fast enough for v=24.
    # 346k blocks * ~76 iterations * (bitwise AND + bit_count) is ~26M ops.
    # Python int bit_count is very fast.
    
    for _ in range(num_triples + 10): # Safety bound
        if uncovered_mask == 0:
            break
        
        best_block_idx = -1
        best_gain = 0
        
        # Iterate through all blocks
        # Optimization: Skip blocks that are already used
        for i, mask in enumerate(block_bitsets):
            if i in used_block_indices:
                continue
            
            # Calculate gain: number of newly covered triples
            # gain = popcount(uncovered_mask & mask)
            common = uncovered_mask & mask
            gain = common.bit_count()
            
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                
        if best_block_idx == -1:
            # Should not happen if cover exists
            break
            
        # Select the block
        selected_blocks.append(all_blocks[best_block_idx])
        used_block_indices.add(best_block_idx)
        
        # Update uncovered mask
        uncovered_mask &= ~block_bitsets[best_block_idx]
        
        # Early termination if no progress (should not happen)
        if best_gain == 0:
            break

    # 4. Post-processing: Redundancy Removal
    # Convert selected blocks to a list of indices for easier manipulation
    # We need to check if removing a block leaves any triple uncovered.
    
    # Build coverage counts for the selected blocks
    cover_count = np.zeros(num_triples, dtype=np.int32)
    for block in selected_blocks:
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            cover_count[idx] += 1
    
    improved_blocks = [list(b) for b in selected_blocks]
    
    # Iteratively remove redundant blocks
    # A block is redundant if all triples it covers have cover_count > 1.
    # We do this in a loop because removing one block might make another redundant.
    
    changed = True
    while changed:
        changed = False
        # Try removing blocks in random order to avoid bias
        indices_to_check = list(range(len(improved_blocks)))
        random.shuffle(indices_to_check)
        
        for idx in indices_to_check:
            block = improved_blocks[idx]
            
            # Check if this block is redundant
            is_redundant = True
            for triple in itertools.combinations(block, t):
                t_idx = triple_to_idx[triple]
                if cover_count[t_idx] <= 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove the block
                # Decrement cover counts
                for triple in itertools.combinations(block, t):
                    t_idx = triple_to_idx[triple]
                    cover_count[t_idx] -= 1
                
                improved_blocks.pop(idx)
                changed = True
                # Adjust indices if necessary, or just break and restart loop
                # Since we are iterating over a shuffled list of indices,
                # and we pop from improved_blocks, the indices in indices_to_check
                # for subsequent iterations in the same while-pass are invalid.
                # It's safer to break and restart the while loop.
                break
        
    return improved_blocks

# EVOLVE-BLOCK-END
