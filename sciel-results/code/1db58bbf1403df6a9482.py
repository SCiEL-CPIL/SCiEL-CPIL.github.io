import itertools
import random

def construct(v, k, t):
    # Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    block_masks = [0] * num_blocks
    blocks_containing_triple = [[] for _ in range(num_triples)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            mask |= (1 << idx)
            blocks_containing_triple[idx].append(b_idx)
        block_masks[b_idx] = mask
        
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple times
    for run in range(100):
        random.seed(run)
        uncovered_list = list(range(num_triples))
        blocks = []
        
        while uncovered_list:
            # Pick random seed
            seed_idx = random.choice(uncovered_list)
            
            best_b_idx = -1
            max_covered = -1
            
            # Find best block containing seed_idx
            for b_idx in blocks_containing_triple[seed_idx]:
                # Count how many from uncovered_list are in this block
                # We don't have a direct mask for uncovered_list, but we have block_masks[b_idx]
                # We need to count intersection of block's triples and uncovered_list.
                # Since uncovered_list is a list of indices, we can just check:
                # How to count efficiently?
                # We can maintain a bitmask `uncovered_mask` alongside `uncovered_list`.
                # Let's add `uncovered_mask`.
                pass
            # Need to re-implement with mask for counting speed
            break
            
    # Re-implementing with mask
    for run in range(100):
        random.seed(run)
        uncovered_mask = (1 << num_triples) - 1
        uncovered_list = list(range(num_triples))
        blocks = []
        
        while uncovered_list:
            seed_idx = random.choice(uncovered_list)
            
            best_b_idx = -1
            max_covered = -1
            candidates = []
            
            for b_idx in blocks_containing_triple[seed_idx]:
                covered = block_masks[b_idx] & uncovered_mask
                count = covered.bit_count()
                if count > max_covered:
                    max_covered = count
                    candidates = [b_idx]
                elif count == max_covered:
                    candidates.append(b_idx)
            
            if not candidates:
                break
                
            chosen_b_idx = random.choice(candidates)
            blocks.append(list(all_blocks[chosen_b_idx]))
            
            # Update uncovered
            new_mask = uncovered_mask & ~block_masks[chosen_b_idx]
            # Update list: keep indices where bit is still set
            # This is O(N)
            uncovered_list = [i for i in uncovered_list if (new_mask >> i) & 1]
            uncovered_mask = new_mask
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    return best_blocks
