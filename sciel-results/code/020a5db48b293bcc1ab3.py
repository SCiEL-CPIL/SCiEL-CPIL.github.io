import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    t_subsets_set = set(all_t_subsets)
    
    # Pre-compute all blocks
    all_blocks = list(itertools.combinations(points, k))
    block_to_t_subsets = {}
    for b in all_blocks:
        # Cache the t-subsets for each block
        block_to_t_subsets[b] = set(itertools.combinations(b, t))
        
    # Pre-compute for each t-subset, which blocks cover it
    # This is needed for the fast greedy update
    t_subset_to_blocks = {ts: [] for ts in all_t_subsets}
    for b in all_blocks:
        for ts in block_to_t_subsets[b]:
            t_subset_to_blocks[ts].append(b)
            
    # Function to remove redundant blocks
    def remove_redundant(blocks):
        blocks_set = set(blocks)
        # Count how many times each t-subset is covered
        cover_count = {ts: 0 for ts in all_t_subsets}
        for b in blocks:
            for ts in block_to_t_subsets[b]:
                cover_count[ts] += 1
                
        new_blocks = []
        for b in blocks:
            # Check if b is redundant
            # A block is redundant if all its t-subsets are covered by other blocks
            # i.e., cover_count[ts] > 1 for all ts in b
            is_redundant = True
            for ts in block_to_t_subsets[b]:
                if cover_count[ts] <= 1:
                    is_redundant = False
                    break
            
            if not is_redundant:
                new_blocks.append(b)
                # We don't update cover_count here because we are just filtering
                # But wait, if we keep b, it contributes to cover_count.
                # If we remove b, we should decrement.
                # This logic is slightly flawed if we modify the set while checking.
                # Better approach:
                # 1. Compute cover_count for all blocks.
                # 2. Iterate through blocks. If cover_count[ts] > 1 for all ts in b, mark for removal.
                # 3. Remove all marked blocks.
                # 4. Repeat until no changes.
                
        # Let's implement the iterative removal properly
        current_blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            cover_count = {ts: 0 for ts in all_t_subsets}
            for b in current_blocks:
                for ts in block_to_t_subsets[b]:
                    cover_count[ts] += 1
                    
            blocks_to_remove = []
            for b in current_blocks:
                is_redundant = True
                for ts in block_to_t_subsets[b]:
                    if cover_count[ts] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    blocks_to_remove.append(b)
                    
            if blocks_to_remove:
                for b in blocks_to_remove:
                    current_blocks.remove(b)
                changed = True
                
        return current_blocks

    def greedy_cover(random_seed):
        random.seed(random_seed)
        uncovered = set(all_t_subsets)
        # current_gain[block] = number of uncovered t-subsets in this block
        current_gain = {b: len(block_to_t_subsets[b]) for b in all_blocks}
        
        selected_blocks = []
        
        while uncovered:
            # Find block with max gain
            # To add randomness, we can pick randomly among top K gains
            # Or just pick max, and if ties, random
            max_gain = max(current_gain.values())
            # Get all blocks with max gain
            candidates = [b for b in all_blocks if current_gain[b] == max_gain]
            if not candidates:
                break # Should not happen if uncovered is not empty
            
            # Pick one randomly to break ties / add diversity
            b = random.choice(candidates)
            selected_blocks.append(b)
            
            # Update gains
            # For each t-subset in b, if it was uncovered, it is now covered.
            # Decrement gain for all blocks that cover this t-subset.
            for ts in block_to_t_subsets[b]:
                if ts in uncovered:
                    uncovered.remove(ts)
                    for b_covering in t_subset_to_blocks[ts]:
                        current_gain[b_covering] -= 1
                        
        return selected_blocks

    best_blocks = None
    best_size = float('inf')
    
    # Run multiple greedy instances with different seeds
    seed = 42
    while time.time() - start_time < time_limit:
        blocks = greedy_cover(seed)
        blocks = remove_redundant(blocks)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            # print(f"New best: {best_size} blocks") # Disabled for eval
            
        seed += 1
        # If we find a very good solution, we might want to try local search on it?
        # For now, just keep restarting.

    return [list(b) for b in best_blocks]
