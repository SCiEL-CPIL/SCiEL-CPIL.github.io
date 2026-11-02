import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Precomputation
    # Generate all t-subsets and map them to indices
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {tset: i for i, tset in enumerate(all_tsets)}
    
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, find the indices of the t-subsets it covers
    t_subsets_per_block = len(list(itertools.combinations(range(k), t)))
    BlockTsets = np.zeros((num_blocks, t_subsets_per_block), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        count = 0
        for sub in itertools.combinations(block, t):
            BlockTsets[i, count] = tset_to_idx[sub]
            count += 1
            
    # 2. Polish function: Remove redundant blocks
    def polish_blocks(blocks):
        if not blocks:
            return blocks
        
        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b in blocks:
            cover_count[BlockTsets[b]] += 1
            
        i = len(blocks) - 1
        while i >= 0:
            b = blocks[i]
            can_remove = True
            for ts in BlockTsets[b]:
                if cover_count[ts] == 1:
                    can_remove = False
                    break
            if can_remove:
                for ts in BlockTsets[b]:
                    cover_count[ts] -= 1
                blocks.pop(i)
            else:
                i -= 1
                
        return blocks
    
    # 3. Pairwise exchange local search
    def local_search_2exchange(blocks, max_iterations=100):
        """Try to improve the solution by swapping one block for another."""
        if not blocks:
            return blocks, False
        
        n = len(blocks)
        if n < 2:
            return blocks, False
        
        # Build coverage count
        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b in blocks:
            cover_count[BlockTsets[b]] += 1
        
        changed = False
        iterations = 0
        
        while iterations < max_iterations and time.time() < deadline:
            iterations += 1
            found_swap = False
            
            # Try random pairs of (block to remove, block to add)
            num_attempts = min(200, n * 50)
            
            for _ in range(num_attempts):
                if time.time() > deadline - 5:
                    break
                    
                # Pick a random block to remove
                remove_idx = random.randint(0, n - 1)
                remove_block = blocks[remove_idx]
                
                # Find t-sets that would become uncovered if we remove this block
                # These are t-sets with cover_count == 1 that are in this block
                must_cover = []
                for ts in BlockTsets[remove_block]:
                    if cover_count[ts] == 1:
                        must_cover.append(ts)
                
                if not must_cover:
                    # This block is redundant, remove it
                    for ts in BlockTsets[remove_block]:
                        cover_count[ts] -= 1
                    blocks.pop(remove_idx)
                    n -= 1
                    changed = True
                    found_swap = True
                    break
                
                # Find a block that covers all must_cover t-sets
                # Use heuristic: look for blocks that share many elements with remove_block
                # and check if they cover all must_cover t-sets
                
                # Get the set of elements in remove_block
                remove_set = set(all_blocks[remove_block])
                
                # Try random blocks, biased toward those that share elements
                for attempt in range(50):
                    if time.time() > deadline - 5:
                        break
                    
                    # Pick a random block, but bias toward blocks that share elements
                    if attempt < 30:
                        # Pick a block that shares at least k-2 elements with remove_block
                        # This is a heuristic to speed up search
                        add_block = random.choice(range(num_blocks))
                    else:
                        add_block = random.randint(0, num_blocks - 1)
                    
                    if add_block == remove_block:
                        continue
                    
                    # Check if add_block covers all must_cover t-sets
                    add_tsets = set(BlockTsets[add_block])
                    if all(ts in add_tsets for ts in must_cover):
                        # Perform the swap
                        for ts in BlockTsets[remove_block]:
                            cover_count[ts] -= 1
                        for ts in BlockTsets[add_block]:
                            cover_count[ts] += 1
                        blocks[remove_idx] = add_block
                        changed = True
                        found_swap = True
                        break
                
                if found_swap:
                    break
            
            if not found_swap:
                break
        
        return blocks, changed
    
    # 4. Greedy Search with Restart and Local Search
    best_blocks = None
    best_len = float('inf')
    
    while time.time() < deadline:
        uncovered = np.ones(num_tsets, dtype=bool)
        blocks = []
        
        # Greedy construction with some randomness
        while uncovered.any():
            gains = uncovered[BlockTsets].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            # Add some randomness: pick from top 10% of candidates
            top_pct = max(1, len(candidates) // 10)
            top_candidates = candidates[:top_pct]
            best_block_idx = top_candidates[random.randint(0, len(top_candidates) - 1)]
            blocks.append(best_block_idx)
            uncovered[BlockTsets[best_block_idx]] = False
        
        # Polish
        blocks = polish_blocks(blocks)
        
        # Local search: 2-exchange
        blocks, _ = local_search_2exchange(blocks, max_iterations=50)
        
        # Final polish after local search
        blocks = polish_blocks(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
    
    if best_blocks is None:
        return []
    
    # Final polish pass
    best_blocks = polish_blocks(best_blocks)
    
    result = [list(all_blocks[b]) for b in best_blocks]
    return result
