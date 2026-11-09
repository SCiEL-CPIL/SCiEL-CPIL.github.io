import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using multiple randomized greedy set cover
    restarts, fast redundancy removal, and 1-exchange local search.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Precompute all t-subsets and assign them indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 2. Precompute coverage bitmask for every possible k-block
    block_masks = []
    all_blocks = []
    
    # Generate all k-blocks and their masks
    for block in itertools.combinations(range(v), k):
        all_blocks.append(block)
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    num_blocks = len(block_masks)
    
    # Helper for bit count
    def count_bits(x):
        if hasattr(x, 'bit_count'):
            return x.bit_count()
        return bin(x).count('1')
        
    def remove_redundant(chosen_indices):
        """
        Remove redundant blocks using coverage counts.
        Returns a list of non-redundant block indices.
        """
        if not chosen_indices:
            return []
            
        # Initialize counts
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        
        # Increment counts for all chosen blocks
        for idx in chosen_indices:
            mask = block_masks[idx]
            # Iterate set bits in mask
            m = mask
            while m:
                lsb = m & -m
                idx_bit = lsb.bit_length() - 1
                counts[idx_bit] += 1
                m ^= lsb
                
        # Iterate and remove redundant blocks
        # We need to be careful: removing a block changes counts for other blocks.
        # A block is redundant if all its t-subsets are covered by at least one OTHER block.
        # i.e., for all t_sub in block, counts[t_sub] > 1.
        
        # We can iterate multiple times or just one pass? 
        # One pass might not be enough if removing A makes B redundant, but B was not redundant before A was removed.
        # However, if we remove a block, counts decrease. This can only make MORE blocks redundant, not fewer.
        # So we can just iterate through the list, and if a block is redundant, remove it and update counts.
        
        active_indices = list(chosen_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(active_indices)):
                idx = active_indices[i]
                mask = block_masks[idx]
                
                # Check if redundant
                is_redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx_bit = lsb.bit_length() - 1
                    if counts[idx_bit] <= 1:
                        is_redundant = False
                        break
                    m ^= lsb
                    
                if is_redundant:
                    # Remove block
                    active_indices.pop(i)
                    # Decrement counts
                    m = mask
                    while m:
                        lsb = m & -m
                        idx_bit = lsb.bit_length() - 1
                        counts[idx_bit] -= 1
                        m ^= lsb
                    changed = True
                    break # Restart loop as indices shifted and counts changed
                    
            if time.time() > deadline - 5.0:
                break
                
        return active_indices

    def greedy_randomized(seed):
        """
        Run randomized greedy set cover.
        """
        random.seed(seed)
        current_uncovered = (1 << num_t_subsets) - 1
        chosen_block_indices = []
        current_union = 0
        
        while current_uncovered != 0:
            if time.time() > deadline - 5.0:
                break
            
            best_gain = -1
            best_block_idx = -1
            best_tie = -1
            
            # Find block with maximum gain
            # To add randomness, we can collect all blocks with max gain and pick one randomly
            # Or simply add a small random noise to the gain.
            # Let's try: find max gain, then pick randomly from all blocks with that gain.
            
            max_gain = 0
            candidates = []
            
            for i in range(num_blocks):
                intersection = block_masks[i] & current_uncovered
                gain = count_bits(intersection)
                
                if gain > max_gain:
                    max_gain = gain
                    candidates = [i]
                elif gain == max_gain and gain > 0:
                    candidates.append(i)
                    
            if not candidates:
                break
            
            # Pick a random candidate
            best_block_idx = random.choice(candidates)
            
            chosen_block_indices.append(best_block_idx)
            current_uncovered &= ~block_masks[best_block_idx]
            current_union |= block_masks[best_block_idx]
            
        return chosen_block_indices

    def local_search_1_exchange(chosen_indices):
        """
        Try to remove each block and replace its unique coverage with at most one new block.
        """
        n = len(chosen_indices)
        if n < 2:
            return chosen_indices
            
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(chosen_indices)):
                if time.time() > deadline - 2.0:
                    break
                    
                idx_i = chosen_indices[i]
                mask_i = block_masks[idx_i]
                
                # Compute union of all other blocks
                union_others = 0
                for j in range(len(chosen_indices)):
                    if i != j:
                        union_others |= block_masks[chosen_indices[j]]
                
                unique_i = mask_i & ~union_others
                if unique_i == 0:
                    # Redundant - remove
                    chosen_indices.pop(i)
                    improved = True
                    break
                
                # Try to find a single block that covers all of unique_i
                # unique_i must be subset of some block's mask
                best_block_idx = -1
                best_overlap = -1
                unique_count = count_bits(unique_i)
                
                # Check all blocks
                for j in range(num_blocks):
                    if j in chosen_indices:
                        continue
                    # Check if block j covers all of unique_i
                    if (block_masks[j] & unique_i) == unique_i:
                        # Found a replacement
                        # Prefer block with most overlap with existing union (consolidation)
                        overlap = count_bits(block_masks[j] & union_others)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_block_idx = j
                            
                if best_block_idx != -1:
                    # Replace block i with best_block_idx
                    chosen_indices[i] = best_block_idx
                    improved = True
                    break
                    
        return chosen_indices

    # Main Strategy:
    # 1. Run several randomized greedy instances
    # 2. Clean up redundancies
    # 3. Keep the best
    # 4. Apply local search to the best
    
    best_solution = []
    
    # Number of restarts depends on time. 
    # Each greedy run is O(N * NumBlocks) roughly. 
    # N ~ 160, NumBlocks ~ 50k. 160 * 50k = 8M ops. 
    # In Python, this might take 1-2 seconds per run.
    # So we can do ~10-20 runs in 110s.
    
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        seed = 42 + r
        candidates = greedy_randomized(seed)
        if not candidates:
            continue
            
        # Remove redundants
        cleaned = remove_redundant(candidates)
        
        if len(cleaned) < len(best_solution):
            best_solution = cleaned
            
    # If we have a solution, try local search on it
    if best_solution:
        best_solution = local_search_1_exchange(best_solution)
        
    # If local search failed or didn't improve, we still have best_solution
    # If best_solution is empty, it means we failed to cover? 
    # Greedy should always cover if possible.
    
    return [list(all_blocks[idx]) for idx in best_solution]
