import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140

    def time_left():
        return deadline - time.time() > 2

    # 1. Precomputation
    all_t_subs = list(itertools.combinations(range(v), t))
    num_t_subs = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    # Map each k-block to the list of t-subset indices it covers
    block_to_t_subs = [[] for _ in range(num_k_blocks)]
    # Map each t-subset to the list of k-block indices that cover it
    t_sub_to_blocks = [[] for _ in range(num_t_subs)]
    
    for b_idx, block in enumerate(all_k_blocks):
        for t_sub in itertools.combinations(block, t):
            t_idx = t_sub_to_idx[t_sub]
            block_to_t_subs[b_idx].append(t_idx)
            t_sub_to_blocks[t_idx].append(b_idx)

    # 2. Greedy Construction
    # We use a simple greedy approach: always pick the block that covers the most uncovered t-subsets.
    # To speed this up, we can use a priority queue, but for the first pass, we'll just scan.
    # Actually, a common optimization is to pick a random uncovered t-subset, then pick the best block covering it.
    # This is faster than scanning all blocks to find the global max gain.
    
    cover_count = [0] * num_t_subs
    uncovered = set(range(num_t_subs))
    blocks = []
    
    while uncovered and time_left():
        anchor_t_idx = random.choice(list(uncovered))
        candidates = t_sub_to_blocks[anchor_t_idx]
        
        best_block_idx = -1
        best_gain = -1
        
        for b_idx in candidates:
            gain = 0
            for t_idx in block_to_t_subs[b_idx]:
                if cover_count[t_idx] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
        
        if best_block_idx == -1:
            break
            
        blocks.append(all_k_blocks[best_block_idx])
        
        for t_idx in block_to_t_subs[best_block_idx]:
            if cover_count[t_idx] == 0:
                uncovered.discard(t_idx)
            cover_count[t_idx] += 1
            
    best_blocks = list(blocks)
    best_size = len(best_blocks)
    
    # 3. Local Search / Improvement
    # Strategy: Randomly remove a block, then greedily fill the gaps.
    # If the resulting size is smaller, keep it.
    
    current_blocks = list(best_blocks)
    current_size = best_size
    
    # Rebuild cover counts for current_blocks
    def rebuild_counts(blocks_list):
        counts = [0] * num_t_subs
        for block in blocks_list:
            # Need to find index of block in all_k_blocks
            # This is O(1) if we had a map, but we have list.
            # Let's create a map for speed in local search.
            pass
        return counts
    
    # Create a fast lookup: block tuple -> index
    block_to_index = {block: i for i, block in enumerate(all_k_blocks)}
    
    counts = [0] * num_t_subs
    for block in current_blocks:
        b_idx = block_to_index[block]
        for t_idx in block_to_t_subs[b_idx]:
            counts[t_idx] += 1
    
    # Active block indices set for fast membership check
    active_block_indices = set(block_to_index[b] for b in current_blocks)
    
    while time_left():
        if not current_blocks:
            break
        
        rem_idx = random.randint(0, len(current_blocks) - 1)
        rem_block = current_blocks[rem_idx]
        rem_b_idx = block_to_index[rem_block]
        
        # Remove it
        current_blocks.pop(rem_idx)
        active_block_indices.discard(rem_b_idx)
        
        # Decrement counts
        for t_idx in block_to_t_subs[rem_b_idx]:
            counts[t_idx] -= 1
            
        # Identify uncovered t-subsets
        uncovered_t_idxs = []
        for i in range(num_t_subs):
            if counts[i] == 0:
                uncovered_t_idxs.append(i)
        
        if not uncovered_t_idxs:
            # We removed a redundant block!
            current_size -= 1
            if current_size < best_size:
                best_size = current_size
                best_blocks = list(current_blocks)
            # Continue loop to try removing more
            continue
        
        # Try to fill the uncovered t-subsets with new blocks
        new_blocks_added = []
        temp_uncovered = set(uncovered_t_idxs)
        
        while temp_uncovered:
            anchor_t_idx = random.choice(list(temp_uncovered))
            candidates = t_sub_to_blocks[anchor_t_idx]
            
            best_block_idx = -1
            best_gain = -1
            
            for b_idx in candidates:
                # Skip if already in active_block_indices
                if b_idx in active_block_indices:
                    continue
                    
                gain = 0
                for t_idx in block_to_t_subs[b_idx]:
                    if counts[t_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                break
            
            new_block = all_k_blocks[best_block_idx]
            new_blocks_added.append(new_block)
            current_blocks.append(new_block)
            active_block_indices.add(best_block_idx)
            
            for t_idx in block_to_t_subs[best_block_idx]:
                if counts[t_idx] == 0:
                    temp_uncovered.discard(t_idx)
                counts[t_idx] += 1
        
        new_size = len(current_blocks)
        
        if new_size < best_size:
            best_size = new_size
            best_blocks = list(current_blocks)
        else:
            # Revert
            for nb in new_blocks_added:
                current_blocks.remove(nb)
                nb_idx = block_to_index[nb]
                active_block_indices.discard(nb_idx)
                for t_idx in block_to_t_subs[nb_idx]:
                    counts[t_idx] -= 1
            
            # Re-add the removed block
            current_blocks.append(rem_block)
            active_block_indices.add(rem_b_idx)
            for t_idx in block_to_t_subs[rem_b_idx]:
                counts[t_idx] += 1

    return best_blocks
