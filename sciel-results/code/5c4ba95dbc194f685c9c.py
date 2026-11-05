import itertools
import time

def construct():
    points = range(12)
    # Precompute all 5-subsets and map them to indices
    all_5sets = list(itertools.combinations(points, 5))
    n_5sets = len(all_5sets)
    five_set_index = {s: i for i, s in enumerate(all_5sets)}
    
    # Precompute all 6-subsets (blocks)
    all_6sets = list(itertools.combinations(points, 6))
    n_6sets = len(all_6sets)
    
    # For each block, compute the bitmask of 5-subsets it covers
    # And for each 5-set, list the blocks that cover it
    block_masks = [0] * n_6sets
    blocks_by_5set = [[] for _ in range(n_5sets)]
    
    for i, block in enumerate(all_6sets):
        mask = 0
        for j in range(6):
            # Remove j-th element to get a 5-set
            subset = tuple(block[:j] + block[j+1:])
            idx = five_set_index[subset]
            mask |= (1 << idx)
            blocks_by_5set[idx].append(i)
        block_masks[i] = mask
        
    # Precompute for each 5-set, the list of blocks sorted by some heuristic
    # We'll use the number of "conflicts" with other blocks as a tiebreaker
    # For now, just keep the list as is
    
    best_blocks = []
    target_size = 132
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def backtrack(covered_mask, current_blocks):
        nonlocal best_blocks
        if len(current_blocks) == target_size:
            best_blocks = list(current_blocks)
            return True
            
        if time.time() > deadline:
            return False
            
        # Find an uncovered 5-set to force a choice
        # Use MRV heuristic: pick the uncovered 5-set with fewest compatible blocks
        
        available = (~covered_mask) & ((1 << n_5sets) - 1)
        if available == 0:
            return len(current_blocks) == target_size

        # MRV: Find the uncovered 5-set with minimum number of compatible blocks
        # To avoid scanning all 792 5-sets every time, we can use a sampling approach
        # or just pick a few candidates
        
        # For efficiency, let's sample up to 20 uncovered 5-sets and pick the one with fewest options
        # Find some uncovered 5-sets
        sample_indices = []
        temp = available
        max_samples = 20
        while temp and len(sample_indices) < max_samples:
            lowest_bit = temp & -temp
            idx = lowest_bit.bit_length() - 1
            sample_indices.append(idx)
            temp ^= lowest_bit
        
        if not sample_indices:
            return False
            
        # For each sampled 5-set, count compatible blocks
        best_idx = -1
        best_count = float('inf')
        best_candidates = []
        
        for idx_5set in sample_indices:
            candidates = blocks_by_5set[idx_5set]
            count = 0
            for b_idx in candidates:
                mask = block_masks[b_idx]
                if not (mask & covered_mask):
                    count += 1
            if count < best_count:
                best_count = count
                best_idx = idx_5set
                best_candidates = [b_idx for b_idx in candidates if not (block_masks[b_idx] & covered_mask)]
        
        if best_count == 0:
            return False
            
        # Sort candidates: prefer blocks that cover "rare" 5-sets
        # For each candidate block, count how many of its 5-sets have few alternatives
        def candidate_score(b_idx):
            mask = block_masks[b_idx]
            score = 0
            temp = mask
            while temp:
                lowest_bit = temp & -temp
                idx = lowest_bit.bit_length() - 1
                # Count how many blocks cover this 5-set and are compatible
                count = 0
                for other_b in blocks_by_5set[idx]:
                    if not (block_masks[other_b] & covered_mask):
                        count += 1
                score += count
                temp ^= lowest_bit
            return score
        
        best_candidates.sort(key=candidate_score)
        
        for b_idx in best_candidates:
            mask = block_masks[b_idx]
            current_blocks.append(b_idx)
            if backtrack(covered_mask | mask, current_blocks):
                return True
            current_blocks.pop()
            
        return False

    # Symmetry breaking: The first block must contain {0, 1, 2, 3, 4}.
    # There are 7 such blocks: {0,1,2,3,4,x} for x in 5..11
    # In the Witt design, exactly one of these appears.
    # We try each one.
    
    for x in range(5, 12):
        first_block_tuple = (0, 1, 2, 3, 4, x)
        # Find index of this block in all_6sets
        # Since all_6sets is sorted, we can compute the index directly or use bisect
        # For simplicity, use index (it's O(n) but n=924, called 7 times)
        first_block_idx = all_6sets.index(first_block_tuple)
        
        initial_mask = block_masks[first_block_idx]
        current_blocks = [first_block_idx]
        
        if backtrack(initial_mask, current_blocks):
            # Convert indices back to lists
            result = [list(all_6sets[i]) for i in best_blocks]
            return result
            
    # Fallback: return empty or best found
    if best_blocks:
        return [list(all_6sets[i]) for i in best_blocks]
    return []

# EVOLVE-BLOCK-END
